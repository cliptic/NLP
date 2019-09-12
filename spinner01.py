import nltk
import numpy as np 
import pandas as pd 
from bs4 import BeautifulSoup
import csv

data = BeautifulSoup(open('electronics/positive.review').read(), features="html.parser")
data = data.findAll('review_text')

data = "<review_text>{}</review_text>".format(data)

soup = BeautifulSoup(data, "html.parser")
print(soup)
with open('positive_review.csv', 'wb') as f_output:
    csv_output = csv.writer(f_output, delimiter='|')

    for review_text in soup.find_all('<review_text>'):
        csv_output.writerow([td.text for td in review_text.find_all('<review_text>')])