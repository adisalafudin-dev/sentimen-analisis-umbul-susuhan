import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
import pickle
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'processed_data.csv'

def labeling(stars):
    if stars >= 3.5:
        return 'positive'
    else:
        return 'negative'

def casefolding(text):
    lowered_text = text.lower()
    return lowered_text

def filtering(text):
    filtered_text = re.sub(r'[^a-zA-Z]', ' ', text)
    return filtered_text

def token(text):
    tokens = word_tokenize(text)
    return tokens

def stopword_removal(tokens):
    stop_words = set(stopwords.words('indonesian')) 
    filtered_token = [token for token in tokens if token.lower() not in stop_words]
    return filtered_token

def stemming(tokens):
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_tokens_joined = ' '.join(stemmed_tokens)
    return stemmed_tokens_joined
    
def main():
    df = pd.read_csv(INPUT_FILE)
    
    df = df[['text', 'stars']].copy()
    
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.strip() != ""]
    
    df.drop_duplicates(subset="text", inplace=True)
    
    df["status"] = df["stars"].apply(labeling)
    
    df_final = df[["text", "status"]]
    
    df_final['text'] = df_final['text'].apply(casefolding)
    
    df_final['text'] = df_final['text'].apply(filtering)
    
    df_final['text'] = df_final['text'].apply(token)
    
    df_final['text'] = df_final['text'].apply(stopword_removal)
    
    df_final['text'] = df_final['text'].apply(stemming)
    
    print(df_final.isna().sum())
    
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    

if __name__ == "__main__":
    main()