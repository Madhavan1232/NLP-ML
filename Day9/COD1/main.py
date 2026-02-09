import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from nlp_utils import clean_text, split_labels

filename = input("Enter dataset filename (CSV or Excel): ")

file_ext = os.path.splitext(filename)[1].lower()

if file_ext == '.csv':
    df = pd.read_csv(os.path.join(sys.path[0], filename))
elif file_ext in ['.xlsx', '.xls']:
    df = pd.read_excel(os.path.join(sys.path[0], filename))
else:
    print("Unsupported file format")
    sys.exit()

if 'text' not in df.columns:
    print("Column 'text' not found")
    sys.exit()

print("=== Dataset Preview ===")
print(df.head())

df['clean_text'] = df['text'].apply(clean_text)

tfidf_vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

if 'sentiment' in df.columns:
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

if 'binary_sentiment' in df.columns:
    le_binary = LabelEncoder()
    binary_encoded = le_binary.fit_transform(df['binary_sentiment'])
    binary_classifier = MultinomialNB()
    binary_classifier.fit(tfidf_matrix, binary_encoded)
    binary_predictions = le_binary.inverse_transform(binary_classifier.predict(tfidf_matrix[:4]))
    print(f"\n===== Binary Classification =====")
    print(f"Binary Predictions: {binary_predictions}")

if 'sentiment' in df.columns:
    multi_classifier = MultinomialNB()
    multi_classifier.fit(tfidf_matrix, df['sentiment_encoded'])
    multi_predictions = multi_classifier.predict(tfidf_matrix[:4])
    print(f"\n===== Multi-Class Classification =====")
    print(f"Multi-Class Predictions: {multi_predictions}")

if 'emotion_labels' in df.columns:
    df['emotion_list'] = df['emotion_labels'].apply(split_labels)
    mlb = MultiLabelBinarizer()
    emotion_matrix = mlb.fit_transform(df['emotion_list'])
    multi_label_classifier = MultiOutputClassifier(DummyClassifier(strategy='constant', constant=0))
    multi_label_classifier.fit(tfidf_matrix, emotion_matrix)
    multi_label_predictions = multi_label_classifier.predict(tfidf_matrix[:4])
    print(f"\n===== Multi-Label Classification =====")
    print(f"Multi-Label Predictions: {multi_label_predictions}")
    print(f"Classes: {mlb.classes_}")

