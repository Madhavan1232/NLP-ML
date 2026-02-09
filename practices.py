import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys , spacy

nlp = spacy.load('en_core_web_sm')
text = open(os.path.join(sys.path[0] , input())).read()
doc = nlp(text)
print(doc)

tokens = [token.text for token in doc[:20]]

for token in tokens:
    print(token)

print(f"Total number of tokens: {len(doc)}  ")
