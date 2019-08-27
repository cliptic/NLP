# This is a code for a sentiment analysis for Amazon reviews

import nltk
import numpy as np 

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))


positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

#define a custom tokenizer
def my_tokenizer(s):
	s = s.lower()
	# use a tokenizer from nltk, better than just split string
	tokens = nltk.tokenize.word_tokenizer(s)
	#remove short words
	tokens = [t for t in tokens if len(t) > 2]
	# lemmatize (eg.: 'dogs'-> 'dog', etc.)
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	# remove stopwords
	tokens = [t for t in tokens if t not in stopwords]


#maps words with intexes
word_index_map = {}
current_index = 0

for review in positive_reviews:
	tokens = my_tokenizer(review.text)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index = 1