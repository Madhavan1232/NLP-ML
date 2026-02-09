import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.simplefilter(action='ignore')
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
filename = input()

try:
    with open(os.path.join(sys.path[0], filename), 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

print("=== Original Text Sample (First 300 chars) ===")
print(content[:300])
print()

doc = nlp(content)

athlete_matcher = PhraseMatcher(nlp.vocab)
athlete_patterns = [nlp.make_doc(name) for name in ["Sarah Claxton", "Sonia O'Sullivan", "Irina Shevchenko"]]
athlete_matcher.add("ATHLETES", athlete_patterns)

event_matcher = PhraseMatcher(nlp.vocab)
event_patterns = [nlp.make_doc(event) for event in ["European Indoor Championships", "World Cross Country Championships", "London marathon", "Bupa Great Ireland Run"]]
event_matcher.add("EVENTS", event_patterns)

print("=== Matched Athlete Names ===")
athlete_matches = athlete_matcher(doc)
if athlete_matches:
    for match_id, start, end in athlete_matches:
        span = doc[start:end]   
        print(f"- {span.text}")
else:
    print("No athlete names found.")
print()

print("=== Matched Sports Events ===")
event_matches = event_matcher(doc)
if event_matches:
    for match_id, start, end in event_matches:
        span = doc[start:end]
        print(f"- {span.text}")
else:
    print("No sports events found.")
