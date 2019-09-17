# Spinner03 on past words
# has to have the same anguage positions

import nltk
import random
import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll("review_text")

# make a trigram dictionary from html text
trigrams = {}
for review in positive_reviews:
	s = review.text.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	for i in range(len(tokens)-3):
		k = (tokens[i], tokens[i+2], )
		if k not in trigrams:
			trigrams[k] = []
		trigrams[k].append(tokens[i+1])

# probabilities of trigrams:
probabilities = {}
for k, words in iter(trigrams.items()):
	if len(set(words)) > 1:
		n = 0
		d = {}
		for w in words:
			if w not in d:
				d[w] = 0
			d[w] += 1
			n += 1
		for w, c in iter(d.items()):
			d[w] = c / n
		probabilities[k] = d

# random select
def random_sample(d):
	r = random.random()
	cumulative = 0
	for w, p in iter(d.items()):
		cumulative += p
		if r < cumulative:
			return w

def test_spinner():
	review = random.choice(positive_reviews)
	review = review.text.lower()
	print('original text: \n', review)
	tokens = nltk.tokenize.word_tokenize(review)
	for i in range(len(tokens) - 3):
		if random.random() < 0.85:
			k = (tokens[i], tokens[i+2])
			if k in probabilities:
				w = random_sample(probabilities[k])
				if nltk.pos_tag(w) == nltk.pos_tag(tokens[i+1]):
					tokens[i+1] = w
	print("spun: \n")
	print(' '.join(tokens).replace(" :",":").replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace("  ' ", "'"))

				
print(test_spinner())
