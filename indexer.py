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

def postInd(dict):
    out_json={}
    for item in dict:
        i=1
        docs=[]
        for value in dict[item]:
            if i==1:
                id=str(value)
            elif i==2:
                reps=str(value)
            else:
                part=str(value)
                this_docs={}
                this_docs['id']=str(id)
                this_docs['reps']=str(reps)
                this_docs['part']=str(part)
                docs.append(this_docs)
                i=0
            i=i+1
        this_word={}
        this_word['idf']=str(len(docs))
        this_word['docs']=docs
        out_json[item]=this_word
    return out_json

def cf():
    with open('.\corpora\cf\json\json_norm\cf_norm.json') as json_file:
        cf_json = json.load(json_file)
    dict=common(cf_json,'papernum')   
    cf_index=postInd(dict)
    with open('.\indices\cf.json', 'w') as f3:
        f3.write(str(cf_index))
        f3.close()


def moocs():
    with open('.\corpora\moocs\json\json_norm\moocs_norm.json') as json_file:
        moocs_json = json.load(json_file)
    dict=common(moocs_json,'courseid')   
    cf_index=postInd(dict)
    with open('.\indices\moocs.json', 'w') as f3:
        f3.write(str(cf_index))
        f3.close()

def main():
    cf()
    #moocs()

if __name__ == '__main__':
    main()


