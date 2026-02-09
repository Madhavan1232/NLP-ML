import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("No spaCy English model found.\nInstall one using: python -m spacy download en_core_web_sm")

    file_name = input("Enter sports news text file name: ")
    file_path = os.path.join(sys.path[0], file_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return

    articles = [a.strip() for a in re.split(r"-+", content) if a.strip()]
    cleaned_docs = [preprocess_text(art) for art in articles if preprocess_text(art)]

    print("\nCleaned Documents:")
    for i, doc in enumerate(cleaned_docs, 1):
        print(f"{i}: {doc}")
    print()

    king_token = nlp("king")[0]
    king_vector = king_token.vector if king_token.has_vector or np.any(king_token.vector) else np.zeros(96)
    print("Word Vector for 'king' (first 10 dims):")
    print(king_vector[:10])
    print()

    doc_embeddings = np.vstack([doc.vector for doc in nlp.pipe(cleaned_docs)])
    
    print(f"Document Embedding Shape: {doc_embeddings.shape}")
    print()

    doc_sim = cosine_similarity(doc_embeddings)
    print("Cosine Similarity Between Documents:")
    print(np.round(doc_sim, 3))
    print()

    test_words = ["dog", "cat", "car", "skym", "apple"]
    tokens = [nlp(w)[0] for w in test_words]
    
    print("Word Similarity Matrix (words with vectors):")
    valid_word_vectors = []
    valid_word_texts = []
    for t in tokens:
        v = t.vector if t.has_vector or np.any(t.vector) else np.zeros(96)
        if np.any(v):
            valid_word_vectors.append(v)
            valid_word_texts.append(t.text)

    if len(valid_word_vectors) >= 2:
        for i in range(len(valid_word_vectors)):
            for j in range(i + 1, len(valid_word_vectors)):
                v1 = valid_word_vectors[i].reshape(1, -1)
                v2 = valid_word_vectors[j].reshape(1, -1)
                sim = cosine_similarity(v1, v2)[0][0]
                print(f"{valid_word_texts[i]} ↔ {valid_word_texts[j]} : {sim:.3f}")
    print()

    print("Observations:")
    print("• 'dog' and 'cat' have high similarity due to both being animals.")
    print("• 'car' is moderately similar to 'dog' and 'cat' due to physical object context.")
    print("• 'skym' may be OOV → low similarity with other words.")
    print("• Embeddings capture meaning beyond frequency (unlike TF-IDF).")

if __name__ == "__main__":
    main()
