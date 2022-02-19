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


def compare_metrics():  # comparar nuestro rendimiento contra el suyo
    a = 1
    # etc


def load_docs(arg):
    path = '.\corpora'
    path2 = '\json\Results\qrels.json'
    corpus = '\\'+arg.c
    # REF
    with open(path+corpus+path2) as f:
        ref_docs = json.loads(f.read())
    # OURS
    # with open() as f:
    #    our_docs = json.loads(f.read())
    our_docs = None
    # TEACHER
    with open('.\TFIDF_reference_results'+corpus+'_ref_qresults.json') as f:
        teach_docs = json.loads(f.read())
    return pd.json_normalize(ref_docs), pd.json_normalize(teach_docs), our_docs


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


def f_beta(avgPrec, avgRec, beta):
    return (beta**2+1)*(avgPrec*avgRec)/(beta**2*avgPrec+avgRec)


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


def MAP_vector(relDocs, compDocs):
    precision = []
    recall = []
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
        prec = hits/(i+1)
        precision.append(prec)
    return precision


def MAP(vPrecision):
    '''input: todos los vectores con las precisiones en cada instante, output:el valor AP'''
    totalAP = 0
    for i in range(len(vPrecision)):
        ap = 0
        if vPrecision[i]:
            for k in range(len(vPrecision[i])):
                if k == 0:
                    ap = ap + vPrecision[i][k]
                else:
                    if vPrecision[i][k - 1] == vPrecision[i][k]:
                        continue
                    else:
                        ap = ap + vPrecision[i][k]
            ap = ap / len(vPrecision[i])
        totalAP = totalAP + ap
    totalAP = totalAP / len(vPrecision)
    return totalAP


def avg_prec_rec(relDocs, compDocs):
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits/len(compDocs)
    rec = hits/len(relDocs)
    return prec, rec


def write_file(txt):
    f = open("metrics.txt", "a")
    f.write(txt)
    f.close()


def metrics(ref_docs, teach_docs, our_docs):
    vTeachPrec = []
    vOurPrec = []
    vTeachMAP = []
    vOurMAP = []
    ''' estas variables acumulan las precisiones y recalls para poder hacer su promedio'''
    teach_overall_prec = 0
    teach_overall_rec = 0
    our_overall_prec = 0
    our_overall_rec = 0
    for index in ref_docs.index:
        # lista con los documentos relevantes para una query
        relDocs = ref_docs["relevantDocs"][index]
        teachDocs = teach_docs["relevantDocs"][index]
        #ourDocs = our_docs["relevantDocs"][index]
        '''Precision-Recall Normalizado'''
        teachPrec = prec_rec(relDocs, teachDocs)
        vTeachPrec.append(teachPrec)
        #ourPrec = prec_rec(relDocs, ourDocs)
        # vOurPrec.append(OurPrec)
        '''MAP- Mean Average Precision'''
        #ourMAP = MAP_vector(relDocs,ourDocs)
        # vOurMAP.append(ourMAP)
        teachMAP = MAP_vector(relDocs, teachDocs)
        vTeachMAP.append(teachMAP)
        '''Valores totales de recall y precision para Fmeasure'''
        #our_avg_prec,our_avg_rec = avg_prec_rec(relDocs,ourDocs)
        #our_overall_prec = our_overall_prec+our_avg_prec
        #our_overall_rec = our_overall_rec+our_avg_rec
        teach_avg_prec, teach_avg_rec = avg_prec_rec(relDocs, teachDocs)
        teach_overall_prec = teach_overall_prec+teach_avg_prec
        teach_overall_rec = teach_overall_rec+teach_avg_rec
    '''MAP'''
    teachAP = MAP(vTeachMAP)
    #ourAP = MAP(vOurMAP)

    '''Valores promedio de recall y precision para Fmeasure'''
    teach_avg_prec = teach_overall_prec/len(teach_docs)
    teach_avg_rec = teach_avg_rec/len(teach_docs)
    write_file("Teacher's average precision: " + str(teach_avg_prec) +
               "\nTeacher's average recall: " + str(teach_avg_rec))
    #our_avg_prec= our_avg_rec/len(our_docs)
    #our_avg_rec= our_avg_rec/len(our_docs)
    '''F1'''
    #our_F1 = f_beta(our_avg_prec,our_avg_rec)
    teach_F1 = f_beta(teach_avg_prec, teach_avg_rec, 1)
    write_file("\nTeacher's F1 Score: " + str(teach_F1))


def main():
    f = open("metrics.txt", "w")
    f.write("")
    f.close()
    args = get_args()
    ref_docs, teach_docs, our_docs = load_docs(args)
    metrics(ref_docs, teach_docs, our_docs)
    compare_metrics()


if __name__ == '__main__':
    main()
