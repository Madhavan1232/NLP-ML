import os
import sys
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def read_from_file(filename):
    file_path = os.path.join(sys.path[0], filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        documents = [line.strip() for line in lines if line.strip()]
        return documents
    except FileNotFoundError:
        print("File not found")
        sys.exit(1)


def read_from_sentence(sentence, nlp):
    doc = nlp(sentence.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


def preprocess_using_spaCy(documents, nlp):
    clean_docs = []
    for doc_text in documents:
        doc = nlp(doc_text.lower())
        tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
        clean_docs.append(" ".join(tokens))
    return clean_docs


def convert_bow_counter(clean_docs):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(clean_docs)
    return vectorizer, bow_matrix


def prepare_output_text(vectorizer, bow_matrix):
    word_freq = bow_matrix.sum(axis=0)
    vocab = vectorizer.vocabulary_
    
    for word, index in vocab.items():
        freq = int(word_freq[0, index])
        print(f"{word} : {freq}")


def convert_bow_tfidf(clean_docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_docs)
    return vectorizer, tfidf_matrix


def print_tfidf_result(features, idf_values, tfidf_matrix):
    for i, feature in enumerate(features):
        print(f"{feature} : IDF={idf_values[i]:.4f}")
