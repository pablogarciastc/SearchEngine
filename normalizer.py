import json
import glob
import string
from nltk import word_tokenize



def delete_fields(entry):  # deleting unnecesary fields
    entry.pop('recordNum', None)
    entry.pop('acessionNum', None)
    entry.pop('authors', None)
    entry.pop('source', None)
    entry.pop('references', None)
    entry.pop('citations', None)
    entry.pop('recordNum', None)
    return entry


def combiner(path1):  # combine several json files to 1
    combined = "["
    for json_file in glob.glob(path1 + '\*.json'):
        with open(json_file) as infile:
            data = json.loads(infile.read())
            for entry in data:
                entry = delete_fields(entry)
                str = json.dumps(entry)
                combined += str
                combined += ",\n"
    combined = combined[:-2]
    combined += "]"
    with open(path1 + '\json_norm\combined_cf.json', 'w') as f3:
        f3.write(combined)
        f3.close()
    return combined


def ini_variables():
    path1 = '.\corpora\cf\json'
    return path1


def lowercase(combined):
    combined = combined.lower()
    return combined


def depunctuation(word):
    return "".join(
        [char for char in word if char not in string.punctuation])


def step1(combined):
    for entry in combined:
        entry['title'] = depunctuation(entry['title'])
        if 'abstract/extract' in entry:
            entry['abstract/extract'] = depunctuation(entry['abstract/extract'])
        if 'majorsubjects' in entry:
            for x in range(len(entry['majorsubjects'])):
                entry['majorsubjects'][x] = depunctuation(entry['majorsubjects'][x])
        if 'minorsubjects' in entry:  ###convertir la lista a string
            for x in range(len(entry['minorsubjects'])):
                entry['minorsubjects'][x] = depunctuation(entry['minorsubjects'][x])
        if 'description' in entry:
            entry['description'] = depunctuation(entry['description'])
    return combined


def normalize(combined, path1):
    combined = lowercase(combined)
    res = json.loads(combined)
    combined = step1(res)


def cf():
    path1 = ini_variables()
    combined = combiner(path1)
    normalize(combined, path1)


def moocs():
    x = ""


def main():
    cf()
    moocs()


if __name__ == '__main__':
    main()
