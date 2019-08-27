# This is a code for a sentiment analysis for Amazon reviews

import nltk
import numpy as np 

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('../stopwords.txt'))


positive_reviews = BeautifulSoup(open('../electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('../electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

#define a custom tokenizer
def my_tokenizer(s):
	s = s.lower()
	# use a tokenizer from nltk, better than just split string
	tokens = nltk.tokenize.word_tokenize(s)
	#remove short words
	tokens = [t for t in tokens if len(t) > 2]
	# lemmatize (eg.: 'dogs'-> 'dog', etc.)
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	# remove stopwords
	tokens = [t for t in tokens if t not in stopwords]
	return tokens


#maps words with intexes
word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
	tokens = my_tokenizer(review.text)
	positive_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

for review in negative_reviews:
	tokens = my_tokenizer(review.text)
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

def tokens_to_vectors(tokens, label):
	x = np.zeros(len(word_index_map)+1)
	for t in tokens:
		i = word_index_map[t]
		x[i] += 1
	x = x / x.sum()
	x[-1] = label
	return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
	xy = tokens_to_vectors(tokens, 1)
	data[i,:] = xy
	i += 1

for tokens in negative_tokenized:
	xy = tokens_to_vectors(tokens, 0)
	data[i,:] = xy
	i += 1

np.random.shuffle(data)
X = data[:,:-1]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classfication with Logistic regression rate:", model.score(Xtest, Ytest))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(Xtrain, Ytrain)
print("Classfication with K - nearest neighbor rate:", model.score(Xtest, Ytest))

from sklearn.svm import SVC
model = SVC()
model.fit(Xtrain, Ytrain)
print("Classfication with SVC rate:", model.score(Xtest, Ytest))

from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier()
model.fit(Xtrain, Ytrain)
print("Classfication with GaussianProcessClassifier rate:", model.score(Xtest, Ytest))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(Xtrain, Ytrain)
print("Classfication with DecisionTreeClassifier rate:", model.score(Xtest, Ytest))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
model = RandomForestClassifier()
model.fit(Xtrain, Ytrain)
print("Classfication with RandomForestClassifier rate:", model.score(Xtest, Ytest))
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classfication with AdaBoostClassifier rate:", model.score(Xtest, Ytest))


'''

threshold = 1

print("Words with a threshold of +-", threshold)
n=0
for word, index in word_index_map.items():
	weight = model.coef_[0][index]
	if weight > threshold or weight < -threshold:
		print(word, weight)
		n += 1
print("A number of words with a threshold +-", threshold, "is:", n)

'''
