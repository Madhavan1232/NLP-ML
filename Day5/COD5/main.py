import os
import sys
import warnings
import spacy
from nltk.stem import SnowballStemmer

# Suppress warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    # Input file name
    print("Enter text file name: ", end="")
    filename = input()
    
    # Verify file path
    file_path = os.path.join(sys.path[0], filename)
    
    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)
        
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    # 1. Original Text Sample
    print("=== Original Text Sample (First 300 chars) ===")
    print(text[:300])
    print()
    
    # Processing
    doc = nlp(text)
    
    # Filter whitespace tokens
    # Detailed requirements: "valid tokens (excluding whitespace)"
    tokens = [token for token in doc if not token.is_space]
    
    # 2. Total Tokens Count
    print(f"Total Tokens Count: {len(tokens)}")
    print()
    
    # Prepare lists
    lemmas = [token.lemma_ for token in tokens]
    
    stemmer = SnowballStemmer(language='english')
    # Stemming constraint: lowercase before stemming
    stems = [stemmer.stem(token.text.lower()) for token in tokens]
    
    # 3. Lemmatized Sample (First 20 tokens)
    print("=== Lemmatized Sample (First 20 tokens) ===")
    print(lemmas[:20])
    print()
    
    # 4. Word --> Lemma Mapping (First 30 tokens)
    print("Word --> Lemma")
    for token, lemma in zip(tokens[:30], lemmas[:30]):
        print(f"{token.text} --> {lemma}")
    print()
    
    # 5. Stemmed Sample (First 20 tokens)
    print("=== Stemmed Sample (First 20 tokens) ===")
    print(stems[:20])
    print()
    
    # 6. Word --> Stem Mapping (First 30 tokens)
    print("Word --> Stem")
    for token, stem in zip(tokens[:30], stems[:30]):
        print(f"{token.text} --> {stem}")
    print()
    
    # 7. Comparison Table
    print("=== Comparison: Lemmatization vs Stemming ===")
    print("Word\t\tLemma\t\tStem")
    print("-" * 42)
    for token, lemma, stem in zip(tokens[:30], lemmas[:30], stems[:30]):
        print(f"{token.text}\t\t{lemma}\t\t{stem}")
    print()
    
    # 8. Conclusion
    print("Conclusion:")
    print("Lemmatization produces dictionary-based meaningful root words")
    print("Stemming may distort words by chopping suffixes")
    print("For NLP tasks like search, topic modeling, and information retrieval, lemmatization gives better and cleaner output")

if __name__ == "__main__":
    main()
