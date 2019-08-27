import numpy as np
import nltk

'''
snt = "What has happened here? This is a dumb shit stupid idea."
token_sentence = nltk.word_tokenize(snt)
print(nltk.pos_tag(token_sentence))
snt = "Machine learning is great."
token_sentence = nltk.word_tokenize(snt)
print(nltk.pos_tag(token_sentence))'''

snt = "Steve Jobs is the CEO of Apple corp."
token_sentence = nltk.word_tokenize(snt)
print(nltk.pos_tag(token_sentence))
tags = nltk.pos_tag(token_sentence)
print(nltk.ne_chunk(tags))
#draws a tree for sentence structure
nltk.ne_chunk(tags).draw()




from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('wolves'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('wolves'))