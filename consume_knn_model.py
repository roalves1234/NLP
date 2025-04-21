print("> Carregamento das bibliotecas")
import pickle
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

print("> Carregamento do modelo KNN")
with open("modelo_knn.pkl", "rb") as file:
    model = pickle.load(file)

print("> Carregamento do modelo BERTimbau")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

print("> Solicitação do título da notícia ao usuário")
title = input("Informe o título da notícia: ").strip()

print("> Pré-processamento do texto")
processed_title = preprocess_text(title)

print("> Geração do embedding para o título")
embeddings = get_bert_embeddings([processed_title], tokenizer, model)

print("> Previsão da categoria")
predicted_category = model.predict(embeddings)[0]

print(f"A categoria prevista para o título informado é: {predicted_category}")