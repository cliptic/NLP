# Spam detector. Section 3
from __future__ import print_function, division
#from future.utils import iteritems
#from builtins import range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import WordCloud

df = pd.read_csv('large_files/spam.csv', encoding = 'ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df.columns = ['labels', 'data']

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values
X = df["data"].values
print(df)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

'''
# transform with TFIDF
tfidf = TfidfVectorizer(decode_error = 'ignore')
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
'''

# transform with CountVectorizer
countV = CountVectorizer(decode_error = 'ignore')
X_train = countV.fit_transform(X_train)
X_test = countV.transform(X_test)

# model
model = MultinomialNB()
model.fit(X_train, Y_train)
print("Train score:", model.score(X_train, Y_train))
print("Test score:", model.score(X_test, Y_test))

# visualize the data
def visualize(label):
	words = ''
	for msg in df[df['labels'] == label]['data']:
		msg = msg.lower()
		words += msg + ' '
	wordcloud = WordCloud(width = 600, height = 400).generate(words)
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()

visualize('spam')
visualize('ham')

#create a predictions column
X = countV.transform(X)
df['predictions'] = model.predict(X)

# those that should be spam:
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
print("THESE ARE THE SNEAKY SPAM MESSAGES:\n")
for msg in sneaky_spam:
	print(msg)

# things that should not be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
print("THESE ARE THE NOT SPAM BUT CLASSIFIED AS THEY WERE:\n")
for msg in not_actually_spam:
	print(msg)

