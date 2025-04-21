print("\n")
print("> Carregamento das bibliotecas")
import pickle
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

print("> Carregamento do modelo KNN")
with open("modelo_knn.pkl", "rb") as file:
    trained_model = pickle.load(file)
    trained_model = trained_model["classifier"]
    
print("> Carregamento do modelo BERT")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
bert = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

print("> Solicitação do título da notícia ao usuário")
title = input("Informe o título da notícia: ").strip()

print("> Pré-processamento e geração do embedding do título\n")
processed_title = preprocess_text(title)
embeddings = get_bert_embeddings([processed_title], tokenizer, bert)
predicted_category = trained_model.predict(embeddings)[0]

print(f"A categoria prevista para o título informado é: {predicted_category}")