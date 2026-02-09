import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.simplefilter(action='ignore')

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError:
    print("Error: spaCy library not found.")
    print("Install using: pip install spacy")
    sys.exit(1)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("SpaCy model 'en_core_web_sm' not found.")
    print("Install using: python -m spacy download en_core_web_sm")
    sys.exit(1)

from sklearn.feature_extraction.text import CountVectorizer

filename = input("Enter sports news text file name: \n")
filepath = os.path.join(sys.path[0], filename)

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

print("=== Original Text Sample ===")
print(content[:300])
print()

documents = [doc.strip() for doc in content.split('---') if doc.strip()]

preprocessed_docs = []
for doc in documents:
    processed = nlp(doc)
    tokens = [token.text.lower() for token in processed 
              if not token.is_stop and not token.is_punct and not token.is_space]
    preprocessed_docs.append(' '.join(tokens))

preprocessed_text = '\n'.join(preprocessed_docs)

print("=== Preprocessed Text Sample ===")
print(preprocessed_text[:300])
print()

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(preprocessed_docs)

print("=== Bag-of-Words Matrix ===")
print(bow_matrix.toarray())
print()

print("=== Word Frequencies ===")
word_freq = bow_matrix.toarray().sum(axis=0)
vocab = vectorizer.get_feature_names_out()
word_freq_dict = dict(zip(vocab, word_freq))

sorted_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

max_word_len = max(len(word) for word, _ in sorted_words)

for word, freq in sorted_words:
    print(f"{word:<{max_word_len}} : {int(freq)}")
