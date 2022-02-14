from nltk.tokenize import word_tokenize
import json

def addToDict(papernum,value,key,dict):
    if type(value) is list:
        for item in value:
            if item in dict:
                if dict[item][len(dict[item])-1]==key and dict[item][len(dict[item])-3]==str(papernum):
                    dict[item][len(dict[item])-2]=str(int(dict[item][len(dict[item])-2])+1)
                else:
                    dict[item] =dict[item]+[str(papernum),'1',key]
            if item not in dict:
                dict[item] = [str(papernum),'1',key]
    return dict

def common(this_json,paper):
    dict={}
    for item in this_json:
        papernum=item[paper]
        for key, value in item.items():
            if key=='title' or key=='abstract/extract' or key=='majorsubjects' or key=='minorsubjects' or key=='description':
                dict=addToDict(papernum,value,key,dict)
    return dict

def cf():
    with open('.\corpora\cf\json\json_norm\cf_norm.json') as json_file:
        cf_json = json.load(json_file)
    dict=common(cf_json,'papernum')   
    with open('.\indices\cf.json', 'w') as f3:
        f3.write(str(dict))
        f3.close()


def moocs():
    with open('.\corpora\moocs\json\json_norm\moocs_norm.json') as json_file:
        moocs_json = json.load(json_file)
    dict=common(moocs_json,'courseid')   
    with open('.\indices\moocs.json', 'w') as f3:
        f3.write(str(dict))
        f3.close()

def main():
    cf()
    moocs()


if __name__ == '__main__':
    main()


