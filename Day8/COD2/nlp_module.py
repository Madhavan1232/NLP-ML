import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.simplefilter(action='ignore')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def read_from_file(filename):
    file_path = os.path.join(sys.path[0], filename)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
        return documents
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)

def read_from_sentence(sentence, nlp):
    doc = nlp(sentence.lower())
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def preprocess_using_spaCy(documents, nlp):
    clean_docs = []
    for doc_text in documents:
        doc = nlp(doc_text.lower())
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        clean_docs.append(" ".join(tokens))
    return clean_docs

def convert_bow_counter(clean_docs):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(clean_docs)
    return vectorizer, bow_matrix

def prepare_output_text(vectorizer, bow_matrix):
    feature_names = vectorizer.get_feature_names()
    for idx, doc_vector in enumerate(bow_matrix.toarray()):
        print(f"Document {idx + 1}:")
        for i, count in enumerate(doc_vector):
            if count > 0:
                print(f"{feature_names[i]}: {count}")
        print()

def convert_bow_tfidf(clean_docs):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_docs)
    features = tfidf_vectorizer.get_feature_names()
    idf_values = tfidf_vectorizer.idf_
    return features, idf_values, tfidf_matrix

def print_tfidf_result(features, idf_values, tfidf_matrix):
    print("TF-IDF Breakdown (Multiple Documents):")
    print()
    
    for doc_idx in range(tfidf_matrix.shape[0]):
        print(f"Document {doc_idx + 1}:")
        doc_vector = tfidf_matrix[doc_idx].toarray().flatten()
        
        for word_idx, tfidf_score in enumerate(doc_vector):
            if tfidf_score > 0:
                word = features[word_idx]
                idf = idf_values[word_idx]
                tf = tfidf_score / idf
                print(f"{word:<12} | TF: {tf:.4f} | IDF: {idf:.4f} | TF-IDF: {tfidf_score:.4f}")
        print()
