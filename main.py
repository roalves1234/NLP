import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings

## Carregamento do dataset
#data_frame = pandas.read_csv("FolhaArticles.csv", sep="\t")
data_frame = pandas.read_csv("FolhaArticles_thin.csv", sep="\t")

## Pré-processamento do texto
data_frame['processed_text'] = data_frame['Title'].apply(preprocess_text)

## Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    data_frame['processed_text'], data_frame['categories'], test_size=0.2, random_state=42
)

## Carregamento do modelo BERTimbau
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

## Geração de embeddings
X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)

## Treinamento do classificador
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_embeddings, y_train)

## Avaliação do modelo
y_pred = classifier.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))