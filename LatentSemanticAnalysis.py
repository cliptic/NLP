# Dimensionality reduction using truncated SVD (aka LSA).

# This transformer performs linear dimensionality reduction 
# by means of truncated singular value decomposition (SVD). 
# Contrary to PCA, this estimator does not center the data 
# before computing the singular value decomposition. 
# This means it can work with scipy.sparse matrices efficiently.

# In particular, truncated SVD works on term count/tf-idf matrices 
# as returned by the vectorizers in sklearn.feature_extraction.text. 
# In that context, it is known as latent semantic analysis (LSA).

import nltk
import numpy as np 
import matplotlib.pyplot as plt 

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('all_book_titles.txt')]
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth', })

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens

word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
for title in titles:
	try:
		title = title.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
		all_titles.append(title)
		tokens = my_tokenizer(title)
		all_tokens.append(tokens)
		for token in tokens:
			if token not in word_index_map:
				word_index_map[token] = current_index
				current_index += 1
				index_word_map.append(token)
	except:
		pass

def tokens_to_vector(tokens): # unsupervised learning - no labels
    x = np.zeros(len(word_index_map)) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
	X[:,i] = tokens_to_vector(tokens)
	i += 1

svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:,0], Z[:,1])
for i in range(D): #xrange(D) for Python2
	plt.annotate(s=index_word_map[i], xy = (Z[i,0], Z[i,1]))
plt.show()

