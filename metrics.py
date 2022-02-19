import json
import argparse
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import numpy as np
import pandas as pd

from indexer import moocs
from metricas import mAP


def compare_metrics():  # comparar nuestro rendimiento contra el suyo
    print("OUR precision:")
    print("TEACHER precision: ")
    print("OUR recall: ")
    print("TEACHER recall: ")
    # etc


def load_docs(arg):
    path = '.\corpora'
    path2 = '\json\Results\qrels.json'
    corpus = '\\'+arg.c
    # REF
    with open(path+corpus+path2) as f:
        ref_docs = json.loads(f.read())
    # OURS
    #with open() as f:
    #    our_docs = json.loads(f.read())
    our_docs = None
    # TEACHER
    with open('.\TFIDF_reference_results'+corpus+'_ref_qresults.json') as f:
        teach_docs = json.loads(f.read())
    return pd.json_normalize(ref_docs), pd.json_normalize(teach_docs) , our_docs


def get_args():
    parser = argparse.ArgumentParser(description='Paso de parámetros')
    parser.add_argument("-c", dest="c")
    parser.add_argument("-rf", dest="rf")
    return parser.parse_args()


def normalize_precision(precision, recall):  # normalizar a Standard 11-level
    normRec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    normPrec = []
    aux = 0
    for i in range(len(normRec)):
        if aux != (len(recall) - 1):
            if i == 0:  # el primero siempre se almacena
                normPrec.append(np.amax(precision[aux:]))
            else:
                if normRec[i] > recall[aux]:
                    aux += 1
                normPrec.append(np.amax(precision[aux:]))
        else:
            if recall[aux] >= normRec[i]:
                normPrec.append(np.amax(precision[aux:]))
            else:
                normPrec.append(0)
    return normPrec


def prec_rec(relDocs, compDocs):  # los objetivamente relevantes y los a comparar
    precision = []
    recall = []
    hits = 0

    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
        prec = hits/(i+1)
        rec = hits/len(relDocs)
        precision.append(prec)
        recall.append(rec)
    precision = normalize_precision(precision, recall)
    return precision

def MAP():
    print("")


def metrics(ref_docs, teach_docs,our_docs):
    vTeachPrec = []
    for index in ref_docs.index:
        # lista con los documentos relevantes para una query
        relDocs = ref_docs["relevantDocs"][index]
        teachDocs = teach_docs["relevantDocs"][index]
        #ourDocs = our_docs["relevantDocs"][index]
        '''Precision-Recall Normalizado'''
        teachPrec = prec_rec(relDocs, teachDocs)
        vTeachPrec.append(teachPrec)
        #ourPrec = prec_rec(relDocs, OurDocs)
        #vOurPrec.append(OurPrec)
    '''MAP- Mean Average Precision'''
    #oursMAP = MAP(vOurPrec)
    teachMAP = MAP(vTeachPrec)

def main():
    args = get_args()
    ref_docs, teach_docs,our_docs = load_docs(args)
    metrics(ref_docs, teach_docs,our_docs)
    compare_metrics()


if __name__ == '__main__':
    main()
