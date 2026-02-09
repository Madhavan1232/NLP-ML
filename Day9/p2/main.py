import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from nltk.corpus import stopwords

warnings.simplefilter("ignore")
nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def split_labels(labels):
    if pd.isna(labels) or labels == '':
        return []
    return [label.strip() for label in str(labels).split(',')]

def split_data(df, test_ratio=0.2, random_state=42):
    test_size = int(len(df) * test_ratio)
    test_df = df.sample(n=test_size, random_state=random_state)
    train_df = df.drop(test_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

filename = input("Enter dataset filename (CSV or Excel): ")

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    sys.exit()

if 'review' not in df.columns:
    print("Column 'review' not found — cannot clean text.")
    sys.exit()

print("=== First 5 Rows ===")
print(df.head())

print(f"\nNumber of samples: {len(df)}")

print("\n=== Data Types ===")
print(df.dtypes)

train_df, test_df = split_data(df)
print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

train_df['cleaned_text'] = train_df['review'].apply(clean_text)
test_df['cleaned_text'] = test_df['review'].apply(clean_text)

print("\n=== Sample Cleaned Text ===")
print(train_df[['review', 'cleaned_text']].head())

tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_train = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])
tfidf_test = tfidf_vectorizer.transform(test_df['cleaned_text'])

with open(os.path.join(sys.path[0], 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print(f"\n=== TF-IDF Shapes ===")
print(f"Train: {tfidf_train.shape} Test: {tfidf_test.shape}")

if 'sentiment' in df.columns:
    le = LabelEncoder()
    le.fit(train_df['sentiment'])
    sentiment_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df['sentiment_encoded'] = le.transform(train_df['sentiment'])
    test_df['sentiment_encoded'] = le.transform(test_df['sentiment'])
    
    print(f"\n=== Sentiment Mapping ===")
    print(sentiment_mapping)

if 'emotion_labels' in df.columns:
    train_df['emotion_list'] = train_df['emotion_labels'].apply(split_labels)
    test_df['emotion_list'] = test_df['emotion_labels'].apply(split_labels)
    
    mlb = MultiLabelBinarizer()
    emotion_train = mlb.fit_transform(train_df['emotion_list'])
    emotion_test = mlb.transform(test_df['emotion_list'])
    
    print(f"\n=== Multi-label Classes ===")
    print(mlb.classes_)
    print(f"Multi-label shape (train): {emotion_train.shape}")
