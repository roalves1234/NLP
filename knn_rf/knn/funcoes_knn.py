import pandas as pd
import torch
import nltk
import numpy as np
import os
import pickle
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
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
                             embeddings_dir="embeddings_knn", batch_size=4, 
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
    print("> Calculando estatísticas de categorias do dataset completo")
    categories = []
    for chunk in tqdm(pd.read_csv(csv_path, sep="\t", chunksize=chunk_size, usecols=['categories']), 
                     total=total_chunks, desc="Contando categorias", unit="chunk"):
        categories.extend(chunk['categories'].astype(str).tolist())
    
    category_counts = pd.Series(categories).value_counts()
    
    return category_counts, embedding_files

def train_knn_model(embedding_files, model_output="modelo_knn.pkl", 
                   test_size=0.2, n_neighbors=5, weights='distance', algorithm='auto', 
                   leaf_size=30):
    """
    Treina um modelo KNN usando arquivos de embeddings.
    
    Args:
        embedding_files (list): Lista de arquivos de embeddings
        model_output (str): Caminho para salvar o modelo final
        test_size (float): Proporção para dados de teste
        n_neighbors (int): Número de vizinhos para o KNN
        weights (str): Função de peso ('uniform' ou 'distance')
        algorithm (str): Algoritmo para computar os vizinhos mais próximos
        leaf_size (int): Tamanho da folha para árvores BallTree ou KDTree
        
    Returns:
        tuple: Modelo treinado e relatório de avaliação
    """
    print("> Coletando dados dos embeddings para treinamento do KNN")
    
    # Coletar todas as categorias únicas primeiro
    all_categories = set()
    for file_path in tqdm(embedding_files, desc="Coletando classes"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_categories.update(set(data['categories']))
    
    all_categories = sorted(list(all_categories))
    print(f"> Total de classes únicas identificadas: {len(all_categories)}")
    
    # Separar os arquivos para treino e teste
    num_files = len(embedding_files)
    test_file_index = int(num_files * (1 - test_size))
    test_files = embedding_files[test_file_index:]
    train_files = embedding_files[:test_file_index]
    
    print(f"> Usando {len(test_files)} arquivos para teste e {len(train_files)} para treino")
    
    # Carregar os dados de treinamento
    train_embeddings = []
    train_categories = []
    
    for train_file in tqdm(train_files, desc="Carregando arquivos de treino", unit="arquivo"):
        with open(train_file, 'rb') as f:
            data = pickle.load(f)
            train_embeddings.append(data['embeddings'])
            train_categories.extend(data['categories'])
    
    # Concatenar os embeddings de treinamento
    if train_embeddings:  # Verificar se a lista não está vazia
        train_embeddings = np.vstack(train_embeddings)
    else:
        print("ERRO: Nenhum dado de treinamento carregado!")
        return None, None
    
    print(f"> Dados de treino: {train_embeddings.shape[0]} exemplos, {train_embeddings.shape[1]} dimensões")
    
    # Carregar dados de teste
    test_embeddings = []
    test_categories = []
    
    for test_file in tqdm(test_files, desc="Carregando arquivos de teste", unit="arquivo"):
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
            test_embeddings.append(data['embeddings'])
            test_categories.extend(data['categories'])
    
    # Concatenar os embeddings de teste
    if test_embeddings:
        test_embeddings = np.vstack(test_embeddings)
    else:
        print("AVISO: Nenhum dado de teste disponível. Usando dados de treinamento para teste.")
        test_embeddings = train_embeddings
        test_categories = train_categories
    
    print(f"> Dados de teste: {test_embeddings.shape[0]} exemplos")
    
    # Criar e treinar o classificador KNN
    print(f"> Treinando modelo KNN (n_neighbors={n_neighbors}, weights='{weights}')")
    knn_classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors, 
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=-1  # Usar todos os processadores disponíveis
    )
    
    # Treinar o KNN
    print("> Iniciando treinamento do KNN (pode levar algum tempo...)")
    with tqdm(total=100, desc="Treinando KNN") as pbar:
        knn_classifier.fit(train_embeddings, train_categories)
        pbar.update(100)
    
    # Avaliar o modelo
    print("> Avaliando modelo KNN")
    
    # Usar batches para prever, pois KNN pode consumir muita memória em grandes datasets
    batch_size = 1000
    all_predictions = []
    
    for i in tqdm(range(0, test_embeddings.shape[0], batch_size), desc="Predição em batches"):
        end_idx = min(i + batch_size, test_embeddings.shape[0])
        batch_X = test_embeddings[i:end_idx]
        batch_pred = knn_classifier.predict(batch_X)
        all_predictions.extend(batch_pred)
    
    # Gerar relatório de classificação
    report = classification_report(test_categories, all_predictions, zero_division=0, output_dict=True)
    
    # Salvar o modelo
    print(f"> Salvando modelo KNN em {model_output}")
    with open(model_output, 'wb') as f:
        pickle.dump({
            'classifier': knn_classifier,
            'classes': all_categories,
            'params': {
                'n_neighbors': n_neighbors,
                'weights': weights,
                'algorithm': algorithm,
                'leaf_size': leaf_size
            }
        }, f)
    
    return knn_classifier, report