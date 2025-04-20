import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
import torch
import gc
import os
import warnings
from tqdm import tqdm
from funcoes_knn import preprocess_text, get_bert_embeddings, get_device

# Filtrar avisos específicos
warnings.filterwarnings("ignore", category=UserWarning, message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Precision is ill-defined.*")

# Configurar variável de ambiente para otimizar alocação de memória CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

# Verificando dispositivo disponível
device = get_device()
print(f"> Usando dispositivo: {device}")

print("> Carregamento do dataset")
# Carregar apenas uma amostra para testes - depois aumentar conforme a capacidade
data_frame = pd.read_csv("FolhaArticles.csv", sep="\t")
sample_size = 2000  # Mantendo o tamanho atual da amostra

# Se o dataset for muito grande, use uma amostra
if len(data_frame) > sample_size:
    print(f"> Usando uma amostra de {sample_size} registros para teste")
    data_frame = data_frame.sample(sample_size, random_state=42)

print("> Pré-processamento do texto")
# Usando tqdm para mostrar progresso do pré-processamento
tqdm.pandas(desc="Pré-processando textos")
data_frame['processed_text'] = data_frame['Title'].progress_apply(preprocess_text)

print("> Garantia de que a coluna 'categories' tenha um tipo consistente")
data_frame['categories'] = data_frame['categories'].astype(str)

# Exibir estatísticas das categorias
category_counts = data_frame['categories'].value_counts()
print(f"> Total de categorias distintas: {len(category_counts)}")
print(f"> Top 5 categorias mais frequentes: \n{category_counts.head(5)}")

# Filtrar categorias com número muito baixo de amostras
# O stratify requer pelo menos 2 amostras por classe
min_samples = 2
print(f"> Filtrando categorias com menos de {min_samples} amostras para permitir stratify")
valid_categories = category_counts[category_counts >= min_samples].index
data_frame = data_frame[data_frame['categories'].isin(valid_categories)]
print(f"> Dataset filtrado: {len(data_frame)} registros após remover categorias raras")
print(f"> Número de categorias após filtro: {len(data_frame['categories'].value_counts())}")

print("> Divisão em treino e teste")
X_train, X_test, y_train, y_test = train_test_split(
    data_frame['processed_text'], 
    data_frame['categories'], 
    test_size=0.2, 
    random_state=42, 
    stratify=data_frame['categories']  # Agora é seguro usar stratify após filtrar
)

# Converter para lista apenas quando necessário para economizar memória
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()

print("> Carregamento do modelo BERTimbau")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Usar precisão de 16 bits para economia de memória
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased", torch_dtype=torch.float16)
# Mover modelo para GPU se disponível
model = model.to(device)

# Limpar memória antes de processar
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("> Geração de embeddings de treino")
# get_bert_embeddings agora usa tqdm internamente
X_train_embeddings = get_bert_embeddings(X_train_list, tokenizer, model, device, batch_size=4)

# Limpar memória após processar os dados de treino
del X_train_list
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("> Geração de embeddings de teste")
X_test_embeddings = get_bert_embeddings(X_test_list, tokenizer, model, device, batch_size=4)

# Liberar memória do modelo quando não for mais necessário
del model
del tokenizer
del X_test_list
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("> Treinamento do classificador KNN")
# Configurar KNN
knn_classifier = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    n_jobs=-1  # Usar todos os processadores disponíveis
)

# Mostrar progresso do treinamento
with tqdm(total=100, desc="Treinando o modelo KNN") as pbar:
    knn_classifier.fit(X_train_embeddings, y_train)
    pbar.update(100)

print("> Avaliação do modelo")
# Usar batches para prever, pois KNN pode consumir muita memória
batch_size = 1000
all_predictions = []

for i in tqdm(range(0, X_test_embeddings.shape[0], batch_size), desc="Predição em batches"):
    end_idx = min(i + batch_size, X_test_embeddings.shape[0])
    batch_X = X_test_embeddings[i:end_idx]
    batch_pred = knn_classifier.predict(batch_X)
    all_predictions.extend(batch_pred)

y_pred = all_predictions

# Obter as classes que têm pelo menos um exemplo previsto
classes_with_predictions = set(y_pred)
print(f"> Número de classes previstas pelo modelo: {len(classes_with_predictions)}")

print("> Relatório de classificação: ")
print(classification_report(y_test, y_pred, zero_division=0))

# Salvar o modelo para uso futuro
print("> Salvando o modelo KNN")
import pickle
with open("modelo_knn_sample.pkl", 'wb') as f:
    pickle.dump({
        'classifier': knn_classifier,
        'params': {
            'n_neighbors': 7,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 30
        }
    }, f)

print("> Processo concluído!")