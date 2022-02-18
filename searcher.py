import json
import pandas as pd
import glob
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datetime import datetime
import sys
import io
import numpy as np
import variables


def lowercase(df):
    return df.applymap(lambda s: s.lower() if type(s) == str else s)

def depunctuation(entry):
    return "".join(
        [char for char in entry if char not in string.punctuation])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]

def normalize_row(row, porter, stop_words):# In order each field is depunctuated, tokenizer, stopwords remover and stemmed)
    row = (row.apply(depunctuation)).apply(word_tokenize)
    row = row.apply(lambda x: [porter.stem(y) for y in x])  # Stem every word.
    row = row.apply(lambda words: [word for word in words if word not in stop_words])
    row = row.apply(lemmatize_text)
    return row



def normalize_fields(df):
    stop_words = stopwords.words('english')
    porter = SnowballStemmer(language='english')
    df.fillna('', inplace=True)
    df['word'] = normalize_row(df['word'], porter, stop_words)
    return df


def normalize(df):
    df = lowercase(df)
    df = normalize_fields(df)
    return df

def tf_ind(doc):
    #falta el divisor !!!!!
    if doc['part']=="title":
        return variables.WEIGHT_TITLE*doc['reps']
    elif doc['part']=="abstract/extract":
        return variables.WEIGHT_ABSTRACT*doc['reps']
    elif doc['part']=="majorSubjects":
        return variables.WEIGHT_MAJOR*doc['reps']
    elif doc['part']=="minorSubjects":
        return variables.WEIGHT_MINOR*doc['reps']
    elif doc['part']=="description":
        return variables.WEIGHT_DESCRIP*doc['reps']
    else:
        return 0

def tf(item,item_json,tf_cf):
    tf_docs={} #{"doc1":{"tf":"tf1"},"doc2":{"tf":"tf2"}}
    for doc in item_json['docs']:
        this_tf={}
        if doc['id'] not in tf_docs:
            tf_docs[doc['id']]=tf_ind(doc)
        else:
            tf_docs[doc['id']]=tf_docs[doc['id']]+tf_ind(doc)
    tf_cf[item]=tf_docs
    return tf_cf

def cf(query):
    tf_cf={}
    bm25f_cf={}

    with open('.\indices\cf.json') as f:
        cf_json = json.loads(f.read())
    for ind in query.index: #cada palabra
        if query['word'][ind][0] in cf_json: 
            tf_cf=tf(query['word'][ind][0],cf_json[query['word'][ind][0]],tf_cf)
            print(tf_cf)
            idf=cf_json[query['word'][ind][0]]['idf']
            #hacer aqui el bm25f
    
def moocs(query):
    with open('.\indices\moocs.json') as f:
        moocs_json = json.loads(f.read())

def processquery(query):
    df = pd.Series(query.split(),
              name="word")
    df = df.apply(lambda x: x.encode('ascii', 'ignore').decode('unicode_escape').strip())
    df=df.to_frame()
    df = normalize(df)
    df=df[df['word'].astype(bool)] 
    return df

def main():    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("INIT=", current_time)
    if len(sys.argv) ==5 and sys.argv[1]=="-c" and sys.argv[3]=="-q":
        if sys.argv[2]=="cf":
            query=processquery(sys.argv[4])
            cf(query)
        elif sys.argv[2]=="moocs":
            query=processquery(sys.argv[4])
            moocs(query)
        else:
            print("PARAMETROS INCORRECTOS")
            exit()
    elif len(sys.argv)==4 and sys.argv[1]=="-c" and sys.argv[2]=="-q":
        query=processquery(sys.argv[3])
        cf(query)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("CF-MOOCS=", current_time)
        moocs(query)
    else:
        print("PARAMETROS INCORRECTOS")
        exit()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("FINISH=", current_time)

if __name__ == '__main__':
    main()
