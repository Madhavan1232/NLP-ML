import os
import sys
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

filename = input("Enter text file name: ")

try:
    data = open(os.path.join(sys.path[0], filename)).read()
except FileNotFoundError:
    print("Error: File not found")
    sys.exit(1)

doc = nlp(data)

stop_words = STOP_WORDS.copy()
stop_words.update({"officially", "announced", "present", "run"})
stop_words.difference_update({"hence", "every", "he"})

filtered_tokens = [token.lemma_.lower() for token in doc 
                   if token.lemma_.lower() not in stop_words 
                   and not token.is_punct 
                   and not token.is_space]

print("Filtered Tokens (First 20):")
print(filtered_tokens[:20])

cleaned_text = " ".join(filtered_tokens)
print("\nCleaned Text Sample:")
print(cleaned_text[:200])