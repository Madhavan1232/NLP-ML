import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import spacy
import warnings
warnings.simplefilter(action='ignore')

def solve():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("SpaCy model 'en_core_web_sm' not found. Install it using:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

    filename = input("Enter text file name: ")
    print()

    file_path = os.path.join(sys.path[0], filename)
    if not os.path.exists(file_path):
        sys.exit(1)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("=== Original Text Sample (First 300 chars) ===")
    print(content[:300])
    print()

    print("=== Sentences with PERSON as Main Subject ===")
    print()

    doc = nlp(content)
    for sent in doc.sents:
        found_subject = None
        found_verb = None
        
        for token in sent:
            if token.dep_ == "nsubj":
                is_person = False
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and token.i >= ent.start and token.i < ent.end:
                        is_person = True
                        break
                if is_person:
                    found_subject = token.text
                    found_verb = token.head.text
                    break
        
        if found_subject:
            print(f"Sentence : {sent.text.strip()}")
            print(f"Subject  : {found_subject}")
            print(f"Main Verb: {found_verb}")
            print("-" * 50)

if __name__ == "__main__":
    solve()