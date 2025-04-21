import pickle
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

# Carregar o modelo KNN
with open("modelo_knn.pkl", "rb") as file:
    knn_model = pickle.load(file)

# Carregar o modelo BERTimbau
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Solicitar o título da notícia ao usuário
title = input("Informe o título da notícia: ").strip()

# Pré-processar o texto
processed_title = preprocess_text(title)

# Gerar embeddings para o título
embeddings = get_bert_embeddings([processed_title], tokenizer, model)

# Fazer a previsão da categoria
predicted_category = knn_model.predict(embeddings)[0]

# Exibir a categoria prevista
print(f"A categoria prevista para o título informado é: {predicted_category}")