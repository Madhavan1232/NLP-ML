import os , sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

text = open(os.path.join(sys.path[0] , "Sample.txt")).read()

doc = nlp(text)

print(nlp(text))
print(doc[:20])
print(f"Total number of tokens: {len(doc)}")