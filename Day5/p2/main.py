import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys ,spacy , re
import nltk
from nltk.stem import SnowballStemmer
import warnings
warnings.simplefilter(action='ignore')

nlp = spacy.load("en_core_web_sm")
data = open(os.path.join(sys.path[0] , input("Enter text file name:"))).read()

print("\nOriginal Text Sample:")
doc = nlp(data)
print(data[:300])

clean_tokens = [t for t in doc if not t.is_space]
print(f"\nTotal Tokens Count: {len(clean_tokens)}")

lemmatized_text = [t.lemma_ for t in clean_tokens]
print("\n=== Lemmatized Sample (First 20 tokens) ===")
print(lemmatized_text[:20])

print("\nWord --> Lemma")
for t in clean_tokens[:30]:
    print(f"{t.text} --> {t.lemma_}")

stemmer = SnowballStemmer(language='english')
stems = [stemmer.stem(t.text.lower()) for t in clean_tokens]
print("\n=== Stemmed Sample (First 20 tokens) ===")
print(stems[:20])

print("\nWord --> Stem")
for t in clean_tokens[:30]:
    print(f"{t.text} --> {stemmer.stem(t.text.lower())}")
    
print("\n=== Comparison: Lemmatization vs Stemming ===")
print("Word\t\tLemma\t\tStem")
print("------------------------------------------")
for t , lemma , stem in zip(clean_tokens[:30] , lemmatized_text[:30] , stems[:30]):
    print(f"{t.text}\t\t{lemma}\t\t{stem}")

print("Conclusion:")
print("Lemmatization produces dictionary-based meaningful root words, while stemming may distort words by chopping suffixes. For NLP tasks like search, topic modeling, and information retrieval, lemmatization gives better and cleaner output.") 
    
    
