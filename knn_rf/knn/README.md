# Classificador KNN para Categorização de Textos

Este projeto implementa um classificador K-Nearest Neighbors (KNN) para categorização de textos em português utilizando embeddings gerados pelo modelo BERT. O sistema foi desenvolvido com foco em otimização de recursos computacionais para lidar com grandes volumes de dados.

## Visão Geral do Sistema

O sistema consiste em três componentes principais:

1. **Funções Auxiliares** (`funcoes_knn.py`): Implementa funções para pré-processamento de texto, geração de embeddings BERT, gerenciamento de dispositivos (CPU/GPU) e treinamento do modelo KNN.
2. **Treinamento do Modelo** (`train_knn_model.py`): Script principal para processamento do dataset em chunks, geração de embeddings e treinamento do classificador KNN.
3. **Demonstração** (`main_knn.py`): Script para demonstrar o fluxo completo de processamento e classificação em uma amostra do dataset.

## Tecnologias Utilizadas

- **BERT para Embeddings**: Utiliza o modelo pré-treinado BERTimbau (neuralmind/bert-base-portuguese-cased) específico para português
- **KNN para Classificação**: Implementado utilizando scikit-learn
- **Processamento de Linguagem Natural**: NLTK para pré-processamento de texto
- **Gerenciamento de Dados**: Pandas para manipulação de datasets

## Fluxo de Processamento

1. **Pré-processamento de Texto**:
   - Tokenização utilizando NLTK
   - Remoção de stopwords em português

2. **Geração de Embeddings**:
   - Utilização do BERTimbau para extrair representações vetoriais dos textos
   - Extração do token [CLS] como representação do texto completo

3. **Classificação**:
   - Treinamento de um modelo KNN utilizando os embeddings gerados
   - Predição de categorias para novos textos

## Técnicas de Otimização Implementadas

### 1. Processamento por Chunks

O dataset é processado em chunks de tamanho configurável (`CHUNK_SIZE`), permitindo o processamento de datasets muito maiores do que a memória disponível:

```python
for chunk in pd.read_csv(CSV_PATH, sep="\t", chunksize=CHUNK_SIZE):
    # Processamento do chunk
```

### 2. Gerenciamento de Recursos GPU

- **Detecção Automática de Dispositivo**: O sistema verifica e utiliza GPU quando disponível
  ```python
  def get_device():
      if torch.cuda.is_available():
          return torch.device("cuda")
      else:
          return torch.device("cpu")
  ```

- **Otimização de Memória CUDA**: Configuração específica para gerenciamento de memória CUDA
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
  ```

- **Limpeza de Cache após processamento**:
  ```python
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  gc.collect()
  ```

### 3. Processamento em Batches

Implementação de processamento em batches para geração de embeddings e predição:

```python
def get_bert_embeddings(texts, tokenizer, model, device, batch_size=4):
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Processamento do batch
```

### 4. Precisão de 16 bits

Utilização de ponto flutuante de 16 bits para reduzir o consumo de memória:

```python
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased", torch_dtype=torch.float16)
```

### 5. Avaliação em Batches

Implementação de avaliação em batches para evitar estouro de memória durante a predição:

```python
for i in range(0, X_test_embeddings.shape[0], batch_size):
    end_idx = min(i + batch_size, X_test_embeddings.shape[0])
    batch_X = X_test_embeddings[i:end_idx]
    batch_pred = knn_classifier.predict(batch_X)
```

### 6. Persistência de Embeddings

Os embeddings gerados são salvos em disco para reutilização, evitando o reprocessamento:

```python
with open(output_file, 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'categories': chunk['categories'].values
    }, f)
```

### 7. Filtragem de Categorias

Implementação de filtragem para categorias com número mínimo de amostras:

```python
category_counts = pd.Series(categories).value_counts()
valid_categories = category_counts[category_counts >= MIN_CATEGORY_COUNT].index.tolist()
```

### 8. Paralelização

Utilização de todos os núcleos disponíveis para treinamento do KNN:

```python
knn_classifier = KNeighborsClassifier(
    n_neighbors=n_neighbors, 
    weights=weights,
    algorithm=algorithm,
    leaf_size=leaf_size,
    n_jobs=-1  # Usar todos os processadores disponíveis
)
```

## Parâmetros de Configuração

- **CHUNK_SIZE**: Tamanho dos chunks para processamento do dataset
- **BATCH_SIZE**: Tamanho do batch para processamento BERT
- **MIN_SAMPLES**: Mínimo de amostras por categoria
- **MIN_CATEGORY_COUNT**: Mínimo de amostras por categoria no dataset completo
- **N_NEIGHBORS**: Número de vizinhos para o KNN
- **WEIGHTS**: Função de peso para o KNN ('uniform' ou 'distance')
- **ALGORITHM**: Algoritmo para computação de vizinhos mais próximos
- **LEAF_SIZE**: Tamanho da folha para árvores BallTree ou KDTree

## Como Executar

1. **Treinamento do Modelo Completo**:
   ```
   python train_knn_model.py
   ```
   Este script processa o dataset completo, gera embeddings e treina o modelo KNN.

2. **Demonstração com Amostra**:
   ```
   python main_knn.py
   ```
   Este script demonstra o fluxo completo em uma amostra menor do dataset.

## Considerações sobre Performance

- O sistema foi projetado para equilibrar uso de memória e velocidade de processamento
- Em GPUs com memória limitada, reduzir `BATCH_SIZE` e `CHUNK_SIZE`
- Para conjuntos de dados muito grandes, aumentar `CHUNK_SIZE` para reduzir o overhead de I/O
- Ajustar `N_NEIGHBORS` conforme a distribuição de classes no dataset

## Estrutura de Arquivos

```
knn/
├── funcoes_knn.py          # Funções auxiliares para processamento e KNN
├── train_knn_model.py      # Script principal para treinar o modelo
├── main_knn.py             # Script de demonstração
├── modelo_knn.pkl          # Modelo treinado (gerado pelo script)
└── embeddings_knn/         # Diretório de embeddings persistidos
    └── embeddings_chunk_*.pkl
```

## Limitações e Possíveis Melhorias

1. **Exploração de Hiperparâmetros**: Implementar busca em grade para otimização de hiperparâmetros do KNN
2. **Visualização de Dados**: Adicionar componentes para visualizar a distribuição das categorias
3. **Balanceamento de Classes**: Implementar técnicas de balanceamento para categorias desbalanceadas
4. **Outras Métricas de Distância**: Explorar diferentes métricas para o KNN além das padrão
5. **Redução de Dimensionalidade**: Investigar PCA ou t-SNE para reduzir dimensionalidade dos embeddings


## Kaggle

https://www.kaggle.com/code/rafaelcalassara/ufg-llm-ex02-option09?scriptVersionId=235311555