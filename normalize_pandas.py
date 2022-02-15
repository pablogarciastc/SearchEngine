import json
import pandas as pd
import glob
import string
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def delete_fields_cf(df):  # deleting unnecesary fields
    del df["recordNum"]
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



def normalize_row(row, porter, stop_words):
    row = (row.apply(depunctuation)).apply(word_tokenize)
    row.apply(lambda words: ' '.join(
        word for word in words if word not in stop_words))
    row = row.apply(lambda x: [porter.stem(y) for y in x])  # Stem every word.
    return row


# In order each field is depunctuated, tokenizer, stopwords remover and stemmed)
def normalize_fields(df):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    df.fillna('', inplace=True)
    if len(df.columns) == 5:  # cf
        df['title'] = normalize_row(df['title'],porter, stop_words)
        df['majorSubjects'] = normalize_row(
            df['majorSubjects'],porter, stop_words)
        df['minorSubjects'] = normalize_row(
            df['minorSubjects'],porter, stop_words)
        df['abstract/extract'] = normalize_row(
            df['abstract/extract'], porter,stop_words)
    if len(df.columns) == 3:  # moocs
        df['title'] = normalize_row(df['title'], porter,stop_words)
        df['description'] = normalize_row(
            df['description'], porter,stop_words)

    return df


def normalize(df):
    df = lowercase(df)
    df = normalize_fields(df)
    return df


def cf():
    path1 = '.\corpora\cf\json'
    cf_json = cf_combiner(path1)
    df_cf = pd.json_normalize(cf_json)
    df_cf = delete_fields_cf(df_cf)
    df_cf = normalize(df_cf)
    result = df_cf.to_json(orient="records")
    parsed = json.loads(result)
    parsed = json.dumps(parsed, indent=4)
    with open(path1 + '\json_norm\cf_norm_pandas.json', 'w') as f3:
        f3.write(parsed)
        f3.close()


def moocs():
    path2 = '.\corpora\moocs\json'
    with open(path2+'\moocs.json') as f:
        moocs_json = json.loads(f.read())
    df_moocs = pd.json_normalize(moocs_json)
    df_moocs = delete_fields_moocs(df_moocs)
    df_moocs = normalize(df_moocs)
    result = df_moocs.to_json(orient="records")
    parsed = json.loads(result)
    parsed = json.dumps(parsed, indent=4)
    with open(path2 + '\json_norm\moocs_norm_pandas.json', 'w') as f3:
        f3.write(parsed)
        f3.close()


def main():
    #cf()
    moocs()


if __name__ == '__main__':
    main()
