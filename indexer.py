from lib2to3.pgen2.tokenize import tokenize
import nltk,math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log10,sqrt
import os
from collections import Counter

def ini_variables():
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    corpusroot = '.\corpora\cf' 
    df=Counter()                      
    tfs={}   
    return stemmer,tokenizer,corpusroot,df,tfs                           

def tokenize_docs(corpusroot,tokenizer,stemmer,tfs,df):
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8') #da error con las carpetas porque las intenta abrir como archivos
        doc = file.read()
        file.close()
        doc = doc.lower()                                                #given code for reading files and converting the case
        tokens = tokenizer.tokenize(doc)                                 #tokenizing each document
        sw=stopwords.words('english')
        tokens = [stemmer.stem(token) for token in tokens if token not in sw]               #removing stopwords and performing stemming
        tf=Counter(tokens)
        df+=Counter(list(set(tokens)))
        tfs[filename]=tf.copy()                                          #making a copy of tf into tfs for that filename
        tf.clear()     



def main():
    stemmer,tokenizer,corpusroot,df,tfs = ini_variables()
    tokenize_docs(corpusroot,tokenizer,stemmer,tfs,df)


if __name__ == '__main__':
    main()