import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords

## Configuração inicial
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')
nltk_stopwords = set(stopwords.words('portuguese'))

def preprocess_text(text):
    # Tokenização
    tokens = nltk.word_tokenize(text, language='portuguese')
    # Remoção de stopwords
    tokens = [word for word in tokens if word.lower() not in nltk_stopwords]
    return ' '.join(tokens)

def get_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()
