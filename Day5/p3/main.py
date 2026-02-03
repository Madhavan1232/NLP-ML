import os , sys 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import spacy

nlp = spacy.load("en_core_web_sm")
data = open(os.path.join(sys.path[0] , input())).read()

lines = data.splitlines()
print("First 10 lines from the file:")
for line in lines[:10]:
    print(line)

doc = nlp(data)

tokens = [token.text for token in doc[:20]]
print("\nFirst 20 tokens:")
print(tokens)

print("\nPOS Tagging Output:")
print("Word\tPOS\tTag")
print("-" * 30)

for token in doc:
        print(f"{token.text} \t {token.pos_} \t {token.tag_}")