import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import spacy

nlp = spacy.load("en_core_web_sm")
filename = input("Enter text file name: \n")

try:
    with open(os.path.join(sys.path[0], filename), 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

print("=== Original Text Sample (First 300 chars) ===")
print(content[:300])

desired_labels = {"PERSON", "GPE", "DATE"}
print("=== Named Entities (PERSON, GPE, DATE) ===")

doc = nlp(content)
for ent in doc.ents:
    if ent.label_ in desired_labels:
        print(f"{ent.text} ({ent.label_})")

from collections import Counter
counter = Counter(ent.label_ for ent in doc.ents)
person , gpe , date = counter["PERSON"], counter["GPE"], counter["DATE"]

print("=== Entity Frequency ===")
print(f"PERSON: {person}")
print(f"GPE: {gpe}")
print(f"DATE: {date}")
