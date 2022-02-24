from nltk.tokenize import word_tokenize
import json

def addToDict(papernum,value,key,dict,docs):
    this_docs={}
    if type(value) is list:
        for item in value:
            if item in dict:
                for doc in dict[item]['docs']:
                    if doc['part']==key and doc['id']==str(papernum):
                        doc['reps']=str(int(doc['reps'])+1)
                    else:
                        this_docs['id']=str(papernum)
                        this_docs['reps']='1'
                        this_docs['part']=key
                        docs[item].append(this_docs)
                        this_word={}
                        this_word['idf']=str(len(docs[item]))
                        this_word['docs']=docs[item]
                        dict[item] =this_word
            elif item not in dict:
                    this_docs={}
                    this_docs['id']=str(papernum)
                    this_docs['reps']='1'
                    this_docs['part']=key
                    docs[item]=[this_docs]
                    this_word={}
                    this_word['idf']=str(len(docs[item]))
                    this_word['docs']=docs[item]
                    dict[item] =this_word
    return dict

def common(this_json,paper):
    dict={}
    docs={}
    for item in this_json:
        papernum=item[paper]
        for key, value in item.items():
            if key=='title' or key=='abstract/extract' or key=='majorSubjects' or key=='minorSubjects' or key=='description':
                dict=addToDict(papernum,value,key,dict,docs)
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
    with open('.\corpora\cf\json\json_norm\cf_norm_pandas.json') as json_file:
        cf_json = json.load(json_file)
    dict=common(cf_json,'paperNum')   
    with open('.\indices\cf.json', 'w') as f3:
        f3.write(str(dict))
        f3.close()


def moocs():
    with open('.\corpora\moocs\json\json_norm\moocs_norm_pandas.json') as json_file:
        moocs_json = json.load(json_file)
    dict=common(moocs_json,'courseID')   
    cf_index=postInd(dict)
    with open('.\indices\moocs.json', 'w') as f3:
        f3.write(str(cf_index))
        f3.close()

def main():
    cf()
    #moocs()

if __name__ == '__main__':
    main()