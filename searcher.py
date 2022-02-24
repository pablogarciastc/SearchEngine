import json
import pandas as pd
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datetime import datetime
import sys
import numpy as np
import variables
import statistics


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

def tf_ind(doc,spec_doc,generic_docs):
    ponderador=(1-variables.BC_TITLE)+variables.BC_TITLE*(spec_doc[doc['part']]/generic_docs[doc['part']])
    if doc['part']=="title":
        return variables.WEIGHT_TITLE*int(doc['reps'])/ponderador
    elif doc['part']=="abstract/extract":
        return variables.WEIGHT_ABSTRACT*int(doc['reps'])/ponderador
    elif doc['part']=="majorSubjects":
        return variables.WEIGHT_MAJOR*int(doc['reps'])/ponderador
    elif doc['part']=="minorSubjects":
        return variables.WEIGHT_MINOR*int(doc['reps'])/ponderador
    elif doc['part']=="description":
        return variables.WEIGHT_DESCRIP*int(doc['reps'])/ponderador
    else:
        return 0

def bm25f(item,item_json,tf_json,lens_json,idf,bm25f_json):
    tf_docs={} #{"doc1":"tf1","doc2":"tf2"}
    bm25f_docs={}
    for doc in item_json['docs']:
        if doc['id'] not in tf_docs:
            tf_docs[doc['id']]=tf_ind(doc,lens_json[doc['id']],lens_json['generic'])
        else:
            tf_docs[doc['id']]=tf_docs[doc['id']]+tf_ind(doc,lens_json[doc['id']],lens_json['generic'])
    tf_json[item]=tf_docs
    for doc in tf_docs:
        bm25f_docs[doc]=((variables.K+1)*tf_docs[doc]*idf)/(tf_docs[doc]+variables.K)
    bm25f_json[item]=bm25f_docs
    return tf_json,bm25f_json

def iterate_words(query,words_json,lens_json):
    tf_json={}
    bm25f_json={}
    ranking={}
    for ind in query.index:
        if query['word'][ind][0] in words_json: 
            if query['word'][ind][0] not in bm25f_json:
                idf=float(words_json[query['word'][ind][0]]['idf'])
                tf_json,bm25f_json=bm25f(query['word'][ind][0],words_json[query['word'][ind][0]],tf_json,lens_json,idf,bm25f_json)
            for doc in bm25f_json[query['word'][ind][0]]:
                if doc in ranking:
                    ranking[doc]=ranking[doc]+bm25f_json[query['word'][ind][0]][doc]
                else:
                    ranking[doc]=bm25f_json[query['word'][ind][0]][doc]
    
    ranking_pd = pd.json_normalize(ranking).transpose()
    ranking_pd = ranking_pd.sort_values(0,ascending=False)
    ranking_pd = ranking_pd.loc[ranking_pd[0].to_numpy() > float(np.percentile(ranking_pd[0].to_numpy(), 70))]
    return ranking_pd

def cf(query):
    with open('.\indices\cf.json') as f:
        cf_json = json.loads(f.read())
    
    with open('.\indices\cf_lens.json') as f:
        cf_lens_json = json.loads(f.read())

    return iterate_words(query,cf_json,cf_lens_json)
    
    
def moocs(query):
    with open('.\indices\moocs.json') as f:
        moocs_json = json.loads(f.read())
    
    with open('.\indices\moocs_lens.json') as f:
        moocs_lens_json = json.loads(f.read())

    return iterate_words(query,moocs_json,moocs_lens_json)

def processquery(query):
    df = pd.Series(query.split(),
              name="word")
    df = df.apply(lambda x: x.encode('ascii', 'ignore').decode('unicode_escape').strip())
    df=df.to_frame()
    df = normalize(df)
    df=df[df['word'].astype(bool)] 
    return df

def printResults(ranking,corpus):
    print("The results for the "+corpus+" dataset is:")
    i=1
    for ind in ranking.index:
        print(str(i)+". "+str(ind))
        i=i+1
        if i==10:
            break
    print("Number of relevant documents in "+corpus+"="+str(len(ranking)))

def cf_queries_file(queries_json,file):
    final_file=[]
    for item in queries_json:
        results=[]
        this_query={}
        query=processquery(item['queryText'])
        ranking=cf(query)
        for ind in ranking.index:
            results.append(int(ind))
            if len(results)==50:
                break
        this_query["queryID"]=item['queryID']
        this_query["relevantDocs"]=results
        final_file.append(this_query)
    
    with open(file, 'w') as f3: #'.\corpora\cf\json\cf_results.json'
        f3.write(json.dumps(final_file))
        f3.close()

def moocs_queries_file(queries_json,file):
    ''' input: output:'''
    final_file=[]
    for item in queries_json:
        results=[]
        this_query={}
        query=processquery(item['queryText'])
        ranking=moocs(query)
        for ind in ranking.index:
            results.append(int(ind))
            if len(results)==50:
                break
        this_query["queryID"]=item['queryID']
        this_query["relevantDocs"]=results
        final_file.append(this_query)
    
    with open(file, 'w') as f3: #'.\corpora\moocs\json\moocs_results.json'
        f3.write(json.dumps(final_file))
        f3.close()

def main():    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("INIT=", current_time)
    if len(sys.argv) ==5 and sys.argv[1]=="-c" and sys.argv[3]=="-q":
        if sys.argv[2]=="cf":
            query=processquery(sys.argv[4])
            printResults(cf(query),"cf")
        elif sys.argv[2]=="moocs":
            query=processquery(sys.argv[4])
            printResults(moocs(query),"moocs")
        else:
            print("INCORRECT PARAMETERS")
            exit()
    elif len(sys.argv) ==5 and sys.argv[1]=="-c" and sys.argv[3]=="-rf":
        if sys.argv[2]=="cf":
            with open('.\corpora\cf\json\queries.json') as f:
                queries_json = json.loads(f.read())
            cf_queries_file(queries_json,sys.argv[4])

        elif sys.argv[2]=="moocs":
            with open('.\corpora\moocs\json\queries.json') as f:
                queries_json = json.loads(f.read())
            moocs_queries_file(queries_json,sys.argv[4])
        else:
            print("INCORRECT PARAMETERS")
            exit()
    elif len(sys.argv) ==7 and sys.argv[1]=="-c" and sys.argv[3]=="-qf" and sys.argv[5]=="-rf":
        if sys.argv[2]=="cf":
            with open(sys.argv[4]) as f:
                queries_json = json.loads(f.read())
            cf_queries_file(queries_json,sys.argv[6])

        elif sys.argv[2]=="moocs":
            with open(sys.argv[4]) as f:
                queries_json = json.loads(f.read())
            moocs_queries_file(queries_json,sys.argv[6])
        else:
            print("INCORRECT PARAMETERS")
            exit()
    else:
        print("INCORRECT PARAMETERS")
        exit()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("FINISH=", current_time)

if __name__ == '__main__':
    main()
