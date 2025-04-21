print("> Carregamento das bibliotecas")
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from funcoes import preprocess_text, get_bert_embeddings
from sklearn.neighbors import KNeighborsClassifier

print("> Carregamento do dataset")
data_frame = pandas.read_csv("FolhaArticles.csv", sep="\t") # Baixe o dataset em https://www.kaggle.com/datasets/luisfcaldeira/folha-news-of-the-brazilian-newspaper-2024

print("> Pré-processamento do texto")
data_frame['processed_text'] = data_frame['Title'].apply(preprocess_text)

print("> Garantia de que a coluna 'categories' tenha um tipo consistente")
data_frame['categories'] = data_frame['categories'].astype(str)

print("> Divisão em treino e teste")
X_train, X_test, y_train, y_test = train_test_split( # X = processed_text, y = categories
    data_frame['processed_text'], data_frame['categories'], test_size=0.2, random_state=42
)

print("> Carregamento do modelo BERTimbau")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

print("> Geração de embeddings de treino")
X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
print("> Geração de embeddings de teste")
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)

# Perguntar ao usuário qual classificador usar
classifier_choice = input("Escolha o classificador (1-RandomForest ou 2-KNN): ").strip().lower()

if classifier_choice == "1":
    print("> Usando o classificador RandomForest")
    classifier = RandomForestClassifier(random_state=42)
elif classifier_choice == "2":
    print("> Usando o classificador KNN")
    classifier = KNeighborsClassifier()
else:
    raise ValueError("Classificador inválido. Escolha 'RandomForest' ou 'KNN'.")

print("> Treinamento do classificador")
classifier.fit(X_train_embeddings, y_train)

print("> Avaliação do modelo")
y_pred = classifier.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))