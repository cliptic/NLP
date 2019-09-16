# Separating html format reviews into text instances and triplets of words.

# build a trigram model
# (word | previous word, next word)

import nltk
import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
import csv
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

soup = BeautifulSoup(open('electronics/positive.review').read(), features="html.parser")


# data = data.findAll('review_text')
# partly copied from last code
text_data = []
for i in soup.find_all('review_text'):
	txt = i.get_text() #append messages to list
	txt = txt.lower()
	lst_txt = tokenizer.tokenize(txt)
	# txt = txt.strip()
	# lst_txt = txt.split()
	text_data.append(lst_txt)

# print(text_data[5])


triple_dictionary = {}
count_dictionary = {}
for i in text_data:
	count = 0
	for n in i:
		if count != 0 and count != (len(i)-1):
			if n not in triple_dictionary:
				triple_dictionary[i[count]] = [[i[count-1], i[count+1]]]
				count_dictionary[i[count]] = 1
			else:
				triple_dictionary[i[count]].append([i[count-1], i[count+1]])
				count_dictionary[i[count]] += 1
		count += 1	

print(triple_dictionary["left"])
print(count_dictionary["left"])

# gives a dictionary with a tuple of two surrounding sorted words and a key, 
# which is a nested dictionary that contains the possible middle word
# and it's cuont in the corpus data
probability_dict = {}
tuple_count_dict = {}
for key in triple_dictionary.keys():
	for value in triple_dictionary[key]:
		value.sort()
		value = tuple(value)
		if value not in probability_dict:
			probability_dict[value] = {key:1}
			tuple_count_dict[value] = 1
		else:
			if key not in probability_dict[value].keys():
				probability_dict[value][key] = 1
			else:
				probability_dict[value][key] += 1
			tuple_count_dict[value] += 1

print(probability_dict[('the', 'the')])
print(tuple_count_dict[('the', 'the')])

### Calculate probabilities
for key in probability_dict.keys():
	for value in probability_dict[key]:
		probability_dict[key][value] = probability_dict[key][value] / tuple_count_dict[key]

print(probability_dict[('the', 'the')])
print(tuple_count_dict[('the', 'the')])