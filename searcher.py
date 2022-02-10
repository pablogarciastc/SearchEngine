import nltk,math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log10,sqrt
import os

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
corpusroot = './corpora'