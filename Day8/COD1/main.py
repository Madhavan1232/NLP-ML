import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.simplefilter(action='ignore')

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Install spaCy model using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

from nlp_module import (read_from_file, read_from_sentence, 
                        preprocess_using_spaCy, convert_bow_counter, 
                        prepare_output_text)

# Task 8.1.2 - Process sample sentence
sample_sentence = "LOL LOL slay slay slay queen."
cleaned_sentence = read_from_sentence(sample_sentence, nlp)

print("Cleaned Sentence:")
print(cleaned_sentence)
print()

vectorizer_sent, bow_sent = convert_bow_counter([cleaned_sentence])

print("BoW Word Frequencies (Sentence):")
vocab_sent = vectorizer_sent.vocabulary_
for word, index in vocab_sent.items():
    freq = int(bow_sent[0, index])
    print(f"{word} : {freq}")
print()

# Task 8.1.3 - Process business news file
filename = input()
documents = read_from_file(filename)
clean_docs = preprocess_using_spaCy(documents, nlp)
vectorizer, bow_matrix = convert_bow_counter(clean_docs)

print("BoW Word Frequencies (Business Data):")
prepare_output_text(vectorizer, bow_matrix)
