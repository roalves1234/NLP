import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

print("> Carregamento do dataset")
data_frame = pandas.read_csv("FolhaArticles_half.csv", sep="\t") # Baixe o dataset em https://www.kaggle.com/datasets/luisfcaldeira/folha-news-of-the-brazilian-newspaper-2024

print("> Pré-processamento do texto")
data_frame['processed_text'] = data_frame['Title'].apply(preprocess_text)

print("> Garantia de que a coluna 'categories' tenha um tipo consisstente")
data_frame['categories'] = data_frame['categories'].astype(str)

print("> Divisão em treino e teste")
X_train, X_test, y_train, y_test = train_test_split(
    data_frame['processed_text'], data_frame['categories'], test_size=0.2, random_state=42
)

print("> Carregamento do modelo BERTimbau")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

print("> Geração de embeddings de treino")
X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
print("> Geração de embeddings de teste")
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)

print("> Treinamento do classificador")
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_embeddings, y_train)

print("> Avaliação do modelo")
y_pred = classifier.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))