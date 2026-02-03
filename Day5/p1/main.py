import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import spacy

nlp = spacy.load("en_core_web_sm")

data = open(os.path.join(sys.path[0] , input())).read()
lines = data.splitlines()

print("First 10 lines from the file:")
for line in lines[:10]:
    print(line)

doc = nlp(data)
token = [token.text for token in doc[:20]]

print("First 20 tokens:")
print(token)