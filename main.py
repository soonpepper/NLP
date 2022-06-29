import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download("maxent_ne_chunker")
nltk.download("words")
from nltk.stem import WordNetLemmatizer

h = "It's a dangerous business, Frodo, going out your door."
h = word_tokenize(h)
stop_words = set(stopwords.words('english'))
joe = [word for word in h if word.casefold() not in stop_words]
joe1 = nltk.pos_tag(h)
bruh2 = """
 Chunk: {<.*>+}
        }<JJ>{"""
chunk_parser = nltk.RegexpParser(bruh2)
joe2 = nltk.ne_chunk(joe1)
def extract_joe(text):
    words = word_tokenize(text, language=language)
    joe1 = nltk.pos_tag(words)
    joe2 = nltk.ne_chunk(joe1, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in joe2
        if hasattr(t, "label") and t.label() == "NE"
    )
extract_joe(h)
