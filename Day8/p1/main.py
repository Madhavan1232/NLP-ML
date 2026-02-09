import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pandas as pd
import spacy
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.simplefilter(action='ignore')

def main():
    try:
        filename = input("Enter text file name: ")
        filepath = os.path.join(sys.path[0], filename)
        
        if not os.path.exists(filepath):
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("=== Original Text Sample (First 300 chars) ===")
        print(content[:300])
        print()
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install the spaCy English model using: python -m spacy download en_core_web_sm")
            sys.exit(1)
            
        doc = nlp(content)
        cleaned_tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        
        print("=== Cleaned Text Sample ===")
        print(" ".join(cleaned_tokens[:50]))
        print()
        
        cleaned_text = " ".join(cleaned_tokens)
        documents = [cleaned_text]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        features = vectorizer.get_feature_names()
        
        print("=== TF-IDF Features ===")
        print(list(features))
        print()
        
        print("=== IDF Values ===")
        idf_values = vectorizer.idf_
        for word, idf in zip(features, idf_values):
            print(f"{word:<20} : {idf:.4f}")
        print()
        
        print("=== TF-IDF Matrix ===")
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
        print(df_tfidf.round(4))
        
    except Exception as e:
        pass

if __name__ == "__main__":
    main()