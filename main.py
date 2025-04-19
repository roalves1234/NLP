import pandas as pandas
import numpy as np
import torch
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from funcoes import preprocess_text, get_bert_embeddings

nltk.download('stopwords')
nltk.download('punkt')

# Configuração inicial
nltk_stopwords = set(stopwords.words('portuguese'))

# Carregamento do dataset
dataset_url = "https://www.kaggleusercontent.com/datasets/marlesson/news-of-the-site-folhauol/download"
dataset_path = "news_dataset.csv"
data_frame = pandas.read_csv(dataset_path)

# Pré-processamento do texto
data_frame['processed_text'] = data_frame['text'].apply(preprocess_text)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    data_frame['processed_text'], data_frame['category'], test_size=0.2, random_state=42
)

# Carregamento do modelo BERTimbau
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Geração de embeddings
X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)

# Treinamento do classificador
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_embeddings, y_train)

# Avaliação do modelo
y_pred = classifier.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))