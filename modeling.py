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


data_clean = pd.read_csv("processed_data.csv")

tfidf = TfidfVectorizer()
tfidf_hasil = tfidf.fit_transform(data_clean['text'])

X_train, X_test, y_train, y_test = train_test_split(tfidf_hasil, data_clean["status"], test_size=0.2, random_state=42)

modelSVM = SVC(kernel="linear", probability=True, class_weight="balanced")
modelSVM.fit(X_train, y_train)
predictedSVM = modelSVM.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictedSVM, labels=modelSVM.classes_), display_labels=modelSVM.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, predictedSVM, zero_division=0))

#Cross Validation

scores = cross_val_score(modelSVM, tfidf_hasil, data_clean["status"], cv=10)
print("Hasil Cross Validation", scores)
print("Rata Rata Cross Validation", np.mean(scores))

df_pos = data_clean[data_clean['text']].str.__contains__