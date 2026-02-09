import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.simplefilter(action='ignore')
import nlp_module

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    print("Install spaCy model using:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

filename = input()

documents = nlp_module.read_from_file(filename)

clean_docs = nlp_module.preprocess_using_spaCy(documents, nlp)

features, idf_values, tfidf_matrix = nlp_module.convert_bow_tfidf(clean_docs)

nlp_module.print_tfidf_result(features, idf_values, tfidf_matrix)

print("Key Observations:")
print("- Common words across documents have lower IDF values.")
print("- Rare words receive higher TF-IDF scores.")
print("- TF-IDF highlights important document-specific terms.")
print("- It improves document comparison over simple BoW.")
