import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pandas as pd
import spacy
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter(action='ignore')

def main():
    filename = input("Enter sports news text file name: ")
    filepath = os.path.join(sys.path[0], filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        sys.exit(1)

    print("=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install the spacy model: python -m spacy download en_core_web_sm")
        sys.exit(1)

    articles = [art.strip() for art in content.split('---') if art.strip()]
    
    cleaned_articles = []
    all_cleaned_tokens = []
    
    for article in articles:
        doc = nlp(article.lower())
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        cleaned_articles.append(" ".join(tokens))
        all_cleaned_tokens.extend(tokens)

    print("=== Cleaned Text Sample ===")
    print(all_cleaned_tokens[:50])
    print()

    bow_vec = CountVectorizer()
    bow_matrix = bow_vec.fit_transform(cleaned_articles)

    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(cleaned_articles)
    feature_names = tfidf_vec.get_feature_names()

    print("=== TF-IDF Features ===")
    print(list(feature_names))
    print()

    print("=== IDF Values ===")
    for word, idf in zip(feature_names, tfidf_vec.idf_):
        print(f"{word} : {idf:.4f}")
    print()

    print("=== TF-IDF Matrix ===")
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    print(df_tfidf.round(4))
    print()

    embeddings = np.array([nlp(art).vector for art in cleaned_articles])
    print("=== Word Embedding Vectors ===")
    print(embeddings)
    print()

    print("=== Vector Shapes ===")
    print(f"BoW shape: {bow_matrix.shape}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print()

    print("=== Cosine Similarity (BoW) ===")
    print(cosine_similarity(bow_matrix))
    print()

    print("=== Cosine Similarity (TF-IDF) ===")
    print(cosine_similarity(tfidf_matrix))
    print()

    print("=== Cosine Similarity (Embeddings) ===")
    print(cosine_similarity(embeddings))
    print()

    print("=== Observations ===")
    print("1. Bag-of-Words considers only word frequency.")
    print("2. TF-IDF highlights important words across documents.")
    print("3. Word embeddings capture semantic meaning and context.")
    print("4. Embedding similarity reflects deeper relationships between sports news articles.")

if __name__ == "__main__":
    main()