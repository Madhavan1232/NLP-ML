import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from nlp_utils import clean_text, split_labels, split_data

warnings.simplefilter("ignore")

filename = input("Enter dataset filename (CSV or Excel): ")

if not filename.endswith(('.csv', '.xlsx', '.xls')):
    print("Only CSV or Excel files supported")
    sys.exit(1)

try:
    file_path = os.path.join(sys.path[0], filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")
except Exception as e:
    print("Error loading file")
    sys.exit(1)

print("=== Dataset Preview ===")
print(df.head())

train, test = split_data(df)

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

print("\n===== Binary Classification =====")
binary_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('clf', LogisticRegression(max_iter=1000))
])
binary_pipeline.fit(train["clean_text"], train["binary_sentiment"])
binary_pred = binary_pipeline.predict(test["clean_text"])

print("Binary Predictions:", binary_pred[:10])
print()
print("Binary Accuracy:", accuracy_score(test["binary_sentiment"], binary_pred))
print()
print("Binary Classification Report:")
print(classification_report(test["binary_sentiment"], binary_pred))
print("Binary Confusion Matrix:")
print(confusion_matrix(test["binary_sentiment"], binary_pred))

print("\n===== Multi-Class Classification =====")
le = LabelEncoder()
train["sentiment_encoded"] = le.fit_transform(train["sentiment"])
test["sentiment_encoded"] = le.transform(test["sentiment"])

multiclass_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('clf', MultinomialNB())
])
multiclass_pipeline.fit(train["clean_text"], train["sentiment_encoded"])
multiclass_pred = multiclass_pipeline.predict(test["clean_text"])

print("Multi-Class Predictions:", multiclass_pred[:10])
print()
print("Multi-Class Accuracy:", accuracy_score(test["sentiment_encoded"], multiclass_pred))
print()
print("Multi-Class Classification Report:")
print(classification_report(test["sentiment_encoded"], multiclass_pred))
print("Multi-Class Confusion Matrix:")
print(confusion_matrix(test["sentiment_encoded"], multiclass_pred))

print("\n===== Multi-Label Classification =====")
train["emotion_list"] = train["emotion_labels"].apply(split_labels)
test["emotion_list"] = test["emotion_labels"].apply(split_labels)

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train["emotion_list"])
Y_test = mlb.transform(test["emotion_list"])

multilabel_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
])
multilabel_pipeline.fit(train["clean_text"], Y_train)
multilabel_pred = multilabel_pipeline.predict(test["clean_text"])

print("Multi-Label Predictions (first 5 rows):")
print(multilabel_pred[:5])
print("Classes:", mlb.classes_)
print()

micro_f1 = f1_score(Y_test, multilabel_pred, average="micro", zero_division=0)
macro_f1 = f1_score(Y_test, multilabel_pred, average="macro", zero_division=0)
print("Multi-Label Micro F1 Score:", micro_f1)
print("Multi-Label Macro F1 Score:", macro_f1)
print()
print("Per-Label F1 Scores:")
per_label_f1 = f1_score(Y_test, multilabel_pred, average=None, zero_division=0)
for emotion, score in zip(mlb.classes_, per_label_f1):
    print(f"{emotion}: {score:.4f}")

print("\n========== SUMMARY ==========")
print("Binary Accuracy:", accuracy_score(test["binary_sentiment"], binary_pred))
print("Multi-Class Accuracy:", accuracy_score(test["sentiment_encoded"], multiclass_pred))
print("Multi-Label Micro F1:", micro_f1)
print("Multi-Label Macro F1:", macro_f1)
