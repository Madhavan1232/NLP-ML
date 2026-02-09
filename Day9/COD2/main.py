import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from nlp_utils import clean_text, split_labels

warnings.simplefilter(action='ignore')

filename = input()

if not filename.endswith(('.csv', '.xlsx', '.xls')):
    print("Unsupported file format")

try:
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(sys.path[0], filename))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(os.path.join(sys.path[0], filename))
except Exception as e:
    print("Error loading file")

if 'text' not in df.columns:
    print("Column 'text' not found")

print("=== First 5 Rows ===")
print(df.head())

print("\nNumber of samples: " + str(len(df)))

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

df['clean_text'] = df['text'].apply(clean_text)

print("\n=== Sample Cleaned Text ===")
print(df[['text', 'clean_text']].head())

tfidf_vectorizer = TfidfVectorizer(max_features=2000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

print(f"\nTF-IDF Shape: {tfidf_matrix.shape}")

if 'sentiment' in df.columns:
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
    sentiment_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\nSentiment Classes: {sentiment_mapping}")

if 'emotion_labels' in df.columns:
    df['emotion_list'] = df['emotion_labels'].apply(split_labels)
    mlb = MultiLabelBinarizer()
    emotion_encoded = mlb.fit_transform(df['emotion_list'])
    print(f"\nEmotion Classes: {mlb.classes_}")
    print(f"Emotion Encoding Shape: {emotion_encoded.shape}")
