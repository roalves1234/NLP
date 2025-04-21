print("> Carregamento das bibliotecas")
import pickle
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

print("> Carregar o modelo KNN")
with open("modelo_knn.pkl", "rb") as file:
    knn_model = pickle.load(file)

print("> Carregar o modelo BERTimbau")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

print("> Solicitar o título da notícia ao usuário")
title = input("Informe o título da notícia: ").strip()

print("> Pré-processar o texto")
processed_title = preprocess_text(title)

print("> Gerar embeddings para o título")
embeddings = get_bert_embeddings([processed_title], tokenizer, model)

print("> Fazer a previsão da categoria")
predicted_category = knn_model.predict(embeddings)[0]

print("> Exibir a categoria prevista")
print(f"A categoria prevista para o título informado é: {predicted_category}")