import os , sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+" , "", text)
    text = re.sub(r"@\w+" , "", text)
    text = re.sub(r"[^a-z\s]" , "" , text)

    tokens = text.split()
    cleaned_text = [word for word in tokens if len(word) > 1]
    return " ".join(cleaned_text)


nlp = spacy.load("en_core_web_sm")
df = pd.read_csv(os.path.join(sys.path[0], input("Enter dataset filename (CSV or Excel): ")))

print("\n=== First 5 Rows ===")
print(df.head())

x = df.drop(columns="binary_sentiment")
y = df["binary_sentiment"]

print(f"Number of samples: {x.shape[0]}")

print("=== Data Types ===")
print(df.dtypes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Train: {x_train.shape[0]}, Test: {y_test.shape[0]}")

train_df