# Natural Language Processing projects

### Spam Detection
Vectorizing text messages and training a multinomial Naive Bayes and AdaBoost classifiers to identify text messages as either "spam" or "ham".

### Sentiment analysis
Lemmatizing text and using BeautifulSoup to visualize the most common words used in positive and negative reviews. Training a Logistic Regression with a threshold of 0.5 to identify positive and negative reviews. Calculating and identifying the misclassified reviews.

<a href="url"><img src="https://github.com/cliptic/nlp/blob/master/jpg/Figure_1.png"  width="320" ></a>
<a href="url"><img src="https://github.com/cliptic/nlp/blob/master/jpg/Figure_2.png"  width="320" ></a>

### Dimensionality reductions
Using PCA and LSA to reduce dimensions of tokenized text vectors. PCA is used to sort book title keywords into a two-dimensional vector, which shows the keywords having two main axis - o containing social/historical keywords; other - scientific and data-driven.

### Article spinner
Modifying articles with randomly selected possible words based on the 2nd order Markov's assumption model. The text is imported from html, converted into tokens, and then - into a trigram dictionary, where every two surrounding words have the possible encountered middle words and their counts (later converted to probabilities) to appear in text. 

### Packages for Python used:
- Pandas
- Numpy
- NLTK
- matplotlib
- sklearn
- wordcloud 
- bs4 (BeautifulSoup)
