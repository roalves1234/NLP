import os
import torch
import gc
import warnings
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from funcoes_knn import (
    get_device, 
    process_dataset_in_chunks, 
    train_knn_model
)

# Ignorar avisos específicos
warnings.filterwarnings("ignore", category=UserWarning, message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Precision is ill-defined.*")

# Configurar variável de ambiente para otimizar alocação de memória CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

# Configurações
CSV_PATH = "FolhaArticles.csv"
EMBEDDINGS_DIR = "embeddings_knn"
MODEL_OUTPUT = "modelo_knn.pkl"
CHUNK_SIZE = 5000  # Tamanho de cada chunk do dataset
BATCH_SIZE = 16     # Tamanho do batch para processamento BERT
MIN_SAMPLES = 2    # Mínimo de amostras por categoria
TEST_SIZE = 0.2    # Proporção para teste
MIN_CATEGORY_COUNT = 10  # Mínimo de amostras por categoria no dataset completo

# Parâmetros do KNN
N_NEIGHBORS = 7    # Número de vizinhos para considerar
WEIGHTS = 'distance'  # Função de peso (uniform ou distance)
ALGORITHM = 'auto'    # Algoritmo para computar vizinhos mais próximos
LEAF_SIZE = 30     # Tamanho da folha para árvores BallTree ou KDTree

def main():
    # Verificando dispositivo disponível
    device = get_device()
    print(f"> Usando dispositivo: {device}")
    
    # Criar pasta para embeddings se não existir
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
    
    # Verificar se já foram gerados embeddings
    existing_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith('embeddings_chunk_') and f.endswith('.pkl')]
    
    if len(existing_files) > 0:
        print(f"> Encontrados {len(existing_files)} arquivos de embeddings existentes")
        embedding_files = [os.path.join(EMBEDDINGS_DIR, f) for f in sorted(existing_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        
        # Buscar as categorias no dataset original para estatísticas
        print("> Calculando estatísticas de categorias do dataset")
        categories = []
        for chunk in pd.read_csv(CSV_PATH, sep="\t", chunksize=CHUNK_SIZE, usecols=['categories']):
            categories.extend(chunk['categories'].astype(str).tolist())
        category_counts = pd.Series(categories).value_counts()
    else:
        print("> Carregando o modelo BERTimbau para gerar embeddings")
        tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased", torch_dtype=torch.float16)
        model = model.to(device)
        
        # Primeira passagem: contar categorias para filtrar as muito raras
        print("> Contando categorias para filtro inicial")
        categories = []
        for chunk in pd.read_csv(CSV_PATH, sep="\t", chunksize=CHUNK_SIZE, usecols=['categories']):
            categories.extend(chunk['categories'].astype(str).tolist())
        
        category_counts = pd.Series(categories).value_counts()
        print(f"> Total de categorias: {len(category_counts)}")
        print(f"> Categorias com pelo menos {MIN_CATEGORY_COUNT} amostras: {sum(category_counts >= MIN_CATEGORY_COUNT)}")
        
        # Filtrar categorias com menos amostras que o mínimo definido
        valid_categories = category_counts[category_counts >= MIN_CATEGORY_COUNT].index.tolist()
        print(f"> Usando {len(valid_categories)} categorias para o treinamento")
        
        # Processar o dataset em chunks e gerar embeddings
        _, embedding_files = process_dataset_in_chunks(
            CSV_PATH, 
            CHUNK_SIZE,
            tokenizer, 
            model, 
            device,
            EMBEDDINGS_DIR,
            BATCH_SIZE,
            categories_filter=valid_categories,
            min_samples=MIN_SAMPLES
        )
        
        # Liberar memória do modelo quando não for mais necessário
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Mostrar estatísticas de categorias
    print("\n> Estatísticas das categorias:")
    print(f"  - Total de categorias: {len(category_counts)}")
    print(f"  - Top 10 categorias mais frequentes:")
    for category, count in category_counts.head(10).items():
        print(f"    * {category}: {count} amostras")
    
    # Treinar o modelo KNN com os embeddings gerados
    print(f"\n> Iniciando treinamento do modelo KNN com {len(embedding_files)} arquivos de embeddings")
    classifier, report = train_knn_model(
        embedding_files,
        model_output=MODEL_OUTPUT,
        test_size=TEST_SIZE,
        n_neighbors=N_NEIGHBORS,
        weights=WEIGHTS,
        algorithm=ALGORITHM,
        leaf_size=LEAF_SIZE
    )
    
    # Mostrar relatório
    print("\n> Relatório de classificação:")
    print(f"  - Acurácia: {report['accuracy']:.4f}")
    print(f"  - Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"  - Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")
    print("\n> Top 10 categorias por F1-score:")
    
    # Mostrar métricas para as 10 categorias com melhor F1-score
    f1_scores = {cat: metrics['f1-score'] for cat, metrics in report.items() 
                 if isinstance(metrics, dict) and cat not in ['accuracy', 'macro avg', 'weighted avg']}
    
    top_categories = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for category, f1 in top_categories:
        support = report[category]['support']
        precision = report[category]['precision']
        recall = report[category]['recall']
        print(f"  - {category}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}")
    
    print(f"\n> Modelo KNN salvo em {MODEL_OUTPUT}")
    print("> Treinamento completo!")

if __name__ == "__main__":
    main()