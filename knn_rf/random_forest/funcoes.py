import pandas as pd
import torch
import nltk
import numpy as np
import os
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

## Configuração inicial
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
nltk_stopwords = set(stopwords.words('portuguese'))

# Função para verificar e configurar o dispositivo (CPU ou GPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def preprocess_text(text):
    # Tokenização
    tokens = nltk.word_tokenize(text, language='portuguese')
    # Remoção de stopwords
    tokens = [word for word in tokens if word.lower() not in nltk_stopwords]
    return ' '.join(tokens)

def get_bert_embeddings(texts, tokenizer, model, device=None, batch_size=4):
    """
    Processa textos em batches para evitar erros de memória CUDA.
    
    Args:
        texts (list): Lista de textos para processar
        tokenizer: Tokenizador BERT
        model: Modelo BERT
        device: Dispositivo para processamento (cuda/cpu)
        batch_size (int): Tamanho do lote para processamento
        
    Returns:
        numpy.ndarray: Array com embeddings gerados
    """
    if device is None:
        device = get_device()
    
    # Limpar cache CUDA antes de começar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Divide os textos em batches para processamento
    all_embeddings = []
    
    # Usar tqdm para mostrar progresso com barra
    for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizar o batch atual
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Mover inputs para o dispositivo
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Processar sem calcular gradientes para economizar memória
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extrair embeddings e mover de volta para CPU
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
        all_embeddings.append(embeddings)
        
        # Liberar memória da GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenar todos os embeddings
    return np.vstack(all_embeddings)

def process_dataset_in_chunks(csv_path, chunk_size, tokenizer, model, device, 
                             embeddings_dir="embeddings", batch_size=4, 
                             categories_filter=None, min_samples=2):
    """
    Processa o dataset em chunks e salva os embeddings em disco.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV
        chunk_size (int): Número de registros a serem processados por vez
        tokenizer: Tokenizador BERT
        model: Modelo BERT
        device: Dispositivo para processamento (cuda/cpu)
        embeddings_dir (str): Diretório para salvar os embeddings
        batch_size (int): Tamanho do batch para processamento dentro de cada chunk
        categories_filter (list): Lista opcional de categorias a serem incluídas
        min_samples (int): Número mínimo de amostras por classe
        
    Returns:
        tuple: Contagem de categorias e lista de arquivos de embeddings gerados
    """
    # Criar diretório para embeddings se não existir
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Lista para armazenar os arquivos de embeddings gerados
    embedding_files = []
    
    # Contador para o número do chunk
    chunk_count = 0
    
    # Contar o total de chunks para a barra de progresso
    total_chunks = sum(1 for _ in pd.read_csv(csv_path, sep="\t", chunksize=chunk_size))
    
    # Processamento por chunks com barra de progresso
    print(f"> Processando dataset em {total_chunks} chunks")
    for chunk in tqdm(pd.read_csv(csv_path, sep="\t", chunksize=chunk_size), 
                     total=total_chunks, desc="Processando chunks", unit="chunk"):
        chunk_count += 1
        
        # Pré-processamento do texto
        chunk['processed_text'] = chunk['Title'].apply(preprocess_text)
        chunk['categories'] = chunk['categories'].astype(str)
        
        # Filtrar por categorias específicas se fornecido
        if categories_filter is not None:
            chunk = chunk[chunk['categories'].isin(categories_filter)]
        
        # Verificar se o chunk tem dados suficientes após o filtro
        if len(chunk) == 0:
            tqdm.write(f"  - Chunk {chunk_count} não contém registros das categorias desejadas. Pulando...")
            continue
        
        # Filtragem de categorias raras dentro do chunk atual
        if min_samples > 1:
            category_counts = chunk['categories'].value_counts()
            valid_categories = category_counts[category_counts >= min_samples].index
            chunk = chunk[chunk['categories'].isin(valid_categories)]
            
            # Verificar se há dados suficientes após o filtro
            if len(chunk) < min_samples:
                tqdm.write(f"  - Chunk {chunk_count} não tem amostras suficientes após filtro. Pulando...")
                continue
        
        # Gerando embeddings
        tqdm.write(f"  - Gerando embeddings para {len(chunk)} registros")
        embeddings = get_bert_embeddings(
            chunk['processed_text'].tolist(), 
            tokenizer, 
            model, 
            device,
            batch_size
        )
        
        # Salvar embeddings e categorias
        output_file = os.path.join(embeddings_dir, f"embeddings_chunk_{chunk_count}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'categories': chunk['categories'].values
            }, f)
        
        embedding_files.append(output_file)
        tqdm.write(f"  - Embeddings salvos em {output_file}")
        
        # Liberar memória
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calcular estatísticas de categorias do dataset inteiro
    # Isso requer uma passagem separada pelo dataset
    print("> Calculando estatísticas de categorias do dataset completo")
    categories = []
    for chunk in tqdm(pd.read_csv(csv_path, sep="\t", chunksize=chunk_size, usecols=['categories']), 
                     total=total_chunks, desc="Contando categorias", unit="chunk"):
        categories.extend(chunk['categories'].astype(str).tolist())
    
    category_counts = pd.Series(categories).value_counts()
    
    return category_counts, embedding_files

def train_incremental_model(embedding_files, model_output="model.pkl", 
                           test_size=0.2, random_state=42):
    """
    Treina um modelo incrementalmente usando arquivos de embeddings.
    
    Args:
        embedding_files (list): Lista de arquivos de embeddings
        model_output (str): Caminho para salvar o modelo final
        test_size (float): Proporção para dados de teste
        random_state (int): Semente aleatória
        
    Returns:
        tuple: Modelo treinado e relatório de avaliação
    """
    # Primeiro passo: identificar todas as classes únicas em todos os arquivos
    # Isso é necessário para garantir consistência nas classes durante o treinamento incremental
    print("> Identificando todas as classes únicas em todos os arquivos")
    all_categories = set()
    for file_path in tqdm(embedding_files, desc="Coletando classes"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_categories.update(set(data['categories']))
    
    all_categories = sorted(list(all_categories))
    print(f"> Total de classes únicas identificadas: {len(all_categories)}")
    
    # Inicializar o classificador com classes conhecidas para evitar inconsistência
    classifier = RandomForestClassifier(
        random_state=random_state, 
        n_jobs=-1,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        warm_start=True  # Permite treinamento incremental
    )
    
    # Separar um conjunto para teste
    test_embeddings = None
    test_categories = None
    
    # Determinar qual arquivo será usado para teste
    num_files = len(embedding_files)
    test_file_index = int(num_files * (1 - test_size))
    test_files = embedding_files[test_file_index:]
    train_files = embedding_files[:test_file_index]
    
    print(f"> Usando {len(test_files)} arquivos para teste e {len(train_files)} para treino")
    
    # Carregar dados de teste
    for test_file in tqdm(test_files, desc="Carregando arquivos de teste", unit="arquivo"):
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        
        if test_embeddings is None:
            test_embeddings = data['embeddings']
            test_categories = data['categories']
        else:
            test_embeddings = np.vstack([test_embeddings, data['embeddings']])
            test_categories = np.concatenate([test_categories, data['categories']])
    
    # Garantir que temos pelo menos uma amostra fictícia para cada classe possível
    # Isso inicializa o classificador com todas as classes possíveis
    dummy_X = np.zeros((len(all_categories), test_embeddings.shape[1]))
    dummy_y = np.array(all_categories)
    
    # Inicializar o classificador com todas as classes possíveis
    print("> Inicializando classificador com todas as classes possíveis")
    classifier.fit(dummy_X, dummy_y)
    
    # Agora podemos treinar incrementalmente com os arquivos de treino reais
    print("> Treinando incrementalmente com os dados reais")
    for i, train_file in enumerate(tqdm(train_files, desc="Treinando com arquivos", unit="arquivo")):
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
        
        # Treinar o modelo com este chunk
        classifier.fit(data['embeddings'], data['categories'])
        
        # Liberar memória
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Avaliação final com o conjunto de teste
    print("> Avaliando modelo com dados de teste")
    try:
        y_pred = classifier.predict(test_embeddings)
        report = classification_report(test_categories, y_pred, zero_division=0, output_dict=True)
    except ValueError as e:
        print(f"Erro durante a predição: {str(e)}")
        print("Tentando abordagem alternativa...")
        
        # Abordagem alternativa: usar predict_proba diretamente e converter para classe com maior probabilidade
        proba = np.zeros((test_embeddings.shape[0], len(all_categories)))
        
        # Processar em batches para evitar problemas de memória
        batch_size = 1000
        for i in tqdm(range(0, test_embeddings.shape[0], batch_size), desc="Predição em batches"):
            end_idx = min(i + batch_size, test_embeddings.shape[0])
            batch_X = test_embeddings[i:end_idx]
            
            try:
                # Tentar prever diretamente
                batch_proba = classifier.predict_proba(batch_X)
                proba[i:end_idx, :batch_proba.shape[1]] = batch_proba
            except:
                # Se falhar, usar cada estimador individualmente
                for tree in classifier.estimators_:
                    try:
                        tree_proba = tree.predict_proba(batch_X)
                        # Adicionar contribuição de cada árvore onde possível
                        cols = min(tree_proba.shape[1], proba.shape[1])
                        proba[i:end_idx, :cols] += tree_proba[:, :cols] / len(classifier.estimators_)
                    except:
                        continue
        
        # Converter probabilidades para classes
        y_indices = np.argmax(proba, axis=1)
        y_pred = np.array([all_categories[idx] for idx in y_indices])
        report = classification_report(test_categories, y_pred, zero_division=0, output_dict=True)
    
    # Salvar o modelo final
    print(f"> Salvando modelo final em {model_output}")
    with open(model_output, 'wb') as f:
        pickle.dump({
            'classifier': classifier,
            'classes': all_categories
        }, f)
    
    return classifier, report
