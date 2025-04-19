import pandas as pd
import numpy as np
import torch
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords

def preprocess_text(text):
    """Função para pré-processar o texto."""
    # Tokenização
    tokens = nltk.word_tokenize(text, language='portuguese')
    # Remoção de stopwords
    tokens = [word for word in tokens if word.lower() not in nltk_stopwords]
    return ' '.join(tokens)

def get_bert_embeddings(texts, tokenizer, model):
    """Função para gerar embeddings usando o modelo BERTimbau."""
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()
