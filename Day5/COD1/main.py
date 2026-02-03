import os , sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

with open(os.path.join(sys.path[0] , input()) , "r") as f:
    text = f.read()

doc = nlp(text)
counter = Counter()

for token in doc:
    counter[token.text] += 1


targets = ['the' , 'at' , 'has' , '.']
for target in targets:
    print(f"{target} : {counter[target]}")