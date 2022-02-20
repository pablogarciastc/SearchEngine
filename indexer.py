#!/usr/bin/python
import json
import sys
import getopt
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
import math

def delete_fields_cf(df):  # deleting unnecesary fields
    del df["paperNum"]
    del df["acessionNum"]
    del df["authors"]
    del df["source"]
    del df["references"]
    del df["citations"]
    return df

def delete_fields_moocs(df):  # deleting unnecesary fields
    del df['url']
    return df

def lists_to_str(entry):
    if 'majorSubjects' in entry:
        entry["majorSubjects"] = " ".join(str(x)
                                          for x in entry["majorSubjects"])
    if 'minorSubjects' in entry:
        entry["minorSubjects"] = " ".join(str(x)
                                          for x in entry["minorSubjects"])
    return entry

def cf_combiner(path1):  # combine several json files to 1
    combined = "["
    for json_file in glob.glob(path1 + '\*.json'):
        with open(json_file) as infile:
            data = json.loads(infile.read())
            for entry in data:
                entry = lists_to_str(entry)
                entry_str = json.dumps(entry, ensure_ascii=False)
                combined += entry_str
                combined += ",\n"
    combined = combined[:-2]
    combined += "]"
    return json.loads(combined)

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
    row = row.apply(
        lambda words: [word for word in words if word not in stop_words])
    row = row.apply(lemmatize_text)
    return row


def normalize_fields(df):
    stop_words = stopwords.words('english')
    porter = SnowballStemmer(language='english')
    df.fillna('', inplace=True)
    if len(df.columns) == 5:  # cf
        df['title'] = normalize_row(df['title'], porter, stop_words)
        df['majorSubjects'] = normalize_row(
            df['majorSubjects'], porter, stop_words)
        df['minorSubjects'] = normalize_row(
            df['minorSubjects'], porter, stop_words)
        df['abstract/extract'] = normalize_row(
            df['abstract/extract'], porter, stop_words)
    if len(df.columns) == 3:  # moocs
        df['title'] = normalize_row(df['title'], porter, stop_words)
        df['description'] = normalize_row(
            df['description'], porter, stop_words)
    return df


def normalize(df):
    df = lowercase(df)
    df = normalize_fields(df)
    return df


def utf_chars(df):
    df['description'] = df['description'].apply(lambda x: x.encode(
        'ascii', 'ignore').decode('unicode_escape').strip())
    df['title'] = df['title'].apply(lambda x: x.encode(
        'ascii', 'ignore').decode('unicode_escape').strip())
    return df


def addToDict(papernum, value, key, dict):
    if type(value) is list:
        for item in value:
            if item in dict:
                if dict[item][len(dict[item])-1] == key and dict[item][len(dict[item])-3] == str(papernum):
                    dict[item][len(dict[item]) -
                               2] = str(int(dict[item][len(dict[item])-2])+1)
                else:
                    dict[item] =dict[item]+[str(papernum),"1",key]
            if item not in dict:
                dict[item] = [str(papernum),"1",key]
    return dict


def common(this_json, paper):
    dict = {}
    for item in this_json:
        papernum = item[paper]
        for key, value in item.items():
            if key=='title' or key=='abstract/extract' or key=='majorSubjects' or key=='minorSubjects' or key=='description':
                dict=addToDict(papernum,value,key,dict)
    return dict

def postInd(dict,lenCorpus):
    out_json={}
    for item in dict:
        idDocs={}
        i=1
        docs=[]
        for value in dict[item]:
            if i == 1:
                id = str(value)
            elif i == 2:
                reps = str(value)
            else:
                part=str(value)
                this_docs={}
                this_docs["id"]=str(id)
                if str(id) not in idDocs:
                    idDocs[str(id)]=0
                this_docs["reps"]=str(reps)
                this_docs["part"]=str(part)
                docs.append(this_docs)
                i=0
            i=i+1
        this_word={}
        this_word["idf"]=str(math.log((lenCorpus+1)/len(idDocs),10))
        this_word["docs"]=docs
        out_json[item]=this_word
    return out_json

def cf_index(parsed,lenCorpus):
    dict=common(parsed,"recordNum")   
    cf_index=postInd(dict,lenCorpus)
    with open('.\indices\cf.json', 'w') as f3:
        f3.write(json.dumps(cf_index))
        f3.close()


def moocs_index(parsed,lenCorpus): 
    dict=common(parsed,"courseID")    
    moocs_index=postInd(dict,lenCorpus)
    with open('.\indices\moocs.json', 'w') as f3:
        f3.write(json.dumps(moocs_index))
        f3.close()

def cf():
    path1 = '.\corpora\cf\json'
    cf_json = cf_combiner(path1)
    docs_cf=len(cf_json)
    df_cf = pd.json_normalize(cf_json)
    df_cf = delete_fields_cf(df_cf)
    df_cf = normalize(df_cf)

    info_generic={}
    
    info_generic['title']=0
    info_generic['majorSubjects']=0
    info_generic['minorSubjects']=0
    info_generic['abstract/extract']=0
    info_doc={}
    for ind in df_cf.index: 
        info_section={}
        info_section['abstract/extract']=len(df_cf['abstract/extract'][ind])
        info_section['majorSubjects']=len(df_cf['majorSubjects'][ind])
        info_section['minorSubjects']=len(df_cf['minorSubjects'][ind])
        info_section['title']=len(df_cf['title'][ind])
        info_doc[str(df_cf['recordNum'][ind])]=info_section

        info_generic['title']=info_generic['title']+info_section['title']
        info_generic['majorSubjects']=info_generic['majorSubjects']+info_section['majorSubjects']
        info_generic['minorSubjects']=info_generic['minorSubjects']+info_section['minorSubjects']
        info_generic['abstract/extract']=info_generic['abstract/extract']+info_section['abstract/extract']
    
    info_generic['title']=info_generic['title']/docs_cf
    info_generic['majorSubjects']=info_generic['majorSubjects']/docs_cf
    info_generic['minorSubjects']=info_generic['minorSubjects']/docs_cf
    info_generic['abstract/extract']=info_generic['abstract/extract']/docs_cf
    info_doc['generic']=info_generic
    

    with open('.\indices\cf_lens.json', 'w') as f3:
        f3.write(json.dumps(info_doc))
        f3.close()
    result = df_cf.to_json(orient="records")
    result = json.loads(result)
    cf_index(result,docs_cf)


def moocs():
    path2 = '.\corpora\moocs\json'
    with open(path2+'\moocs.json') as f:
        moocs_json = json.loads(f.read())
    df_moocs = pd.json_normalize(moocs_json)
    docs_moocs=len(df_moocs)
    df_moocs = delete_fields_moocs(df_moocs)
    df_moocs = utf_chars(df_moocs)
    df_moocs = normalize(df_moocs)

    info_generic={}
    
    info_generic['title']=0
    info_generic['description']=0
    info_doc={}
    for ind in df_moocs.index: 
        info_section={}
        info_section['description']=len(df_moocs['description'][ind])
        info_section['title']=len(df_moocs['title'][ind])
        info_doc[str(df_moocs['courseID'][ind])]=info_section

        info_generic['title']=info_generic['title']+info_section['title']
        info_generic['description']=info_generic['description']+info_section['description']
    
    info_generic['title']=info_generic['title']/docs_moocs
    info_generic['description']=info_generic['description']/docs_moocs
    info_doc['generic']=info_generic
    
    with open('.\indices\moocs_lens.json', 'w') as f3:
        f3.write(json.dumps(info_doc))
        f3.close()
    result = df_moocs.to_json(orient="records")
    result = json.loads(result)
    moocs_index(result,docs_moocs)


def main():    
    if len(sys.argv) == 3 and sys.argv[1]=="-c":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if sys.argv[2]=="cf":
            print("INIT=", current_time)
            cf()
        elif sys.argv[2]=="moocs":
            print("INIT=", current_time)
            moocs()
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
