import json
import glob
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def delete_fields_cf(entry):  # deleting unnecesary fields
    entry.pop('recordNum', None)
    entry.pop('acessionNum', None)
    entry.pop('authors', None)
    entry.pop('source', None)
    entry.pop('references', None)
    entry.pop('citations', None)
    entry.pop('recordNum', None)
    return entry

def delete_fields_moocs(entry):  # deleting unnecesary fields
    entry.pop('url', None)
    return entry

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
                entry = delete_fields_cf(entry)
                entry = lists_to_str(entry)
                entry_str = json.dumps(entry)
                combined += entry_str
                combined += ",\n"
    combined = combined[:-2]
    combined += "]"
    return combined

def moocs_str(moocs_json):
    #json list to json str
    json_str = "["
    for entry in moocs_json:
        entry = delete_fields_moocs(entry)
        entry = lists_to_str(entry)
        entry_str = json.dumps(entry)
        json_str += entry_str
        json_str += ",\n"
    json_str = json_str[:-2]
    json_str += "]"
    print(json_str)
    return json_str


def lowercase(combined):
    return combined.lower()


def depunctuation(entry):
    return "".join(
        [char for char in entry if char not in string.punctuation])


def tokenizer(entry):
    return word_tokenize(entry)


def remove_stopwords(entry, stopwords):
    return [word for word in entry if word not in stopwords]


def stemming(entry, porter):
    return [porter.stem(word) for word in entry]


def normalize_fields(combined, path):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    for entry in combined:
        # In order each field is depunctuated, tokenizer, stopwords remover and stemmed)
        entry['title'] = stemming(remove_stopwords(
            tokenizer(depunctuation(entry['title'])), stop_words), porter)
        if 'abstract/extract' in entry:
            entry['abstract/extract'] = stemming(remove_stopwords(tokenizer(depunctuation(
                entry['abstract/extract'])), stop_words), porter)
        if 'majorsubjects' in entry:
            entry['majorsubjects'] = stemming(remove_stopwords(tokenizer(depunctuation(
                entry['majorsubjects'])), stop_words), porter)
        if 'minorsubjects' in entry:  # convertir la lista a string
            entry['minorsubjects'] = stemming(remove_stopwords(tokenizer(depunctuation(
                entry['minorsubjects'])), stop_words), porter)
        if 'description' in entry:
            entry['description'] = stemming(remove_stopwords(tokenizer(
                depunctuation(entry['description'])), stop_words), porter)
    return combined


def normalize(combined, path):
    combined = lowercase(combined)
    combined = normalize_fields(json.loads(combined), path)
    return json.dumps(combined)


def cf():
    path1 =  '.\corpora\cf\json'
    cf_json = cf_combiner(path1)
    cf_json = normalize(cf_json, path1)
    with open(path1 + '\json_norm\cf_norm.json', 'w') as f3:
        f3.write(cf_json)
        f3.close()


def moocs():
    path2 =  '.\corpora\moocs\json'
    with open(path2+'\moocs.json') as json_file:
        moocs_json = json.load(json_file)
    moocs_json = moocs_str(moocs_json)
    moocs_json = normalize(moocs_json,path2)
    with open(path2 + '\json_norm\moocs_norm.json', 'w') as f3:
        f3.write(moocs_json)
        f3.close()


def main():
    cf()
    moocs()


if __name__ == '__main__':
    main()
