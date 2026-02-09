import re
import string
import pandas as pd

ENGLISH_STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by do does doing down during each few for from further had
has have having he her here hers him himself his how i if in into is it its itself me
more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where
which while who whom why with you your yours yourself yourselves
""".split())


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in ENGLISH_STOPWORDS]
    text = ' '.join(words)
    
    return text


def split_labels(label_string):
    if pd.isna(label_string) or label_string == "":
        return []
    
    labels = [label.strip() for label in label_string.split(',')]
    
    return labels


def split_data(df, test_ratio=0.2, random_state=42):
    test_size = int(len(df) * test_ratio)
    test = df.sample(n=test_size, random_state=random_state)
    train = df.drop(test.index)
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, test
