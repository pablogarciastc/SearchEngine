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
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
        prec = hits/(i+1)
        precision.append(prec)
    return precision


def MAP(vPrecision, refDocs):
    '''input: todos los vectores con las precisiones en cada instante, output:el valor AP'''
    totalAP = 0
    for i in range(len(vPrecision)):
        queryAP = 0
        if vPrecision[i]:
            for k in range(len(vPrecision[i])):
                queryAP = queryAP + vPrecision[i][k]
            # divido entre el nº de docs relevantes de esa query
            queryAP = queryAP / len(refDocs["relevantDocs"][i])
        totalAP = totalAP + queryAP
    AP = totalAP / len(vPrecision)
    return AP


def MRR(relDocs, compDocs):
    aux = 1
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            return (1/aux)
        aux = aux+1
    return 0


'''Multilevel Relevance Evaluation'''


def MRE(relDocs, compDocs):

    return 1


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


def p_at_n(relDocs, compDocs, n):
    hits = 0
    if len(compDocs) < n:
        n = len(compDocs)
    for i in range(n):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits/n
    return prec


def rPrecision(relDocs, compDocs):
    hits = 0
    n = len(relDocs)
    if len(compDocs) < n:
        n = len(compDocs)
    for i in range(n):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits/n
    return prec


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
    teachAt5 = 0
    ourAt5 = 0
    teachRPrec = 0
    ourRPrec = 0
    teachMRR = 0
    ourMRR = 0
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
        '''p at N'''
        teachAt5 = teachAt5 + p_at_n(relDocs, teachDocs, 5)
        #our_at_5 = p_at_n(relDocs,ourDocs,5)
        '''r-Precision'''
        teachRPrec = teachRPrec + rPrecision(relDocs, teachDocs)
        '''MRR'''
        teachMRR = teachMRR + MRR(relDocs, teachDocs)
        '''MRE'''
        teachMRE = MRE(relDocs,teachDocs) #en proceso hahahaha
        '''Valores totales de recall y precision para Fmeasure'''
        #our_avg_prec,our_avg_rec = avg_prec_rec(relDocs,ourDocs)
        #our_overall_prec = our_overall_prec+our_avg_prec
        #our_overall_rec = our_overall_rec+our_avg_rec
        teach_avg_prec, teach_avg_rec = avg_prec_rec(relDocs, teachDocs)
        teach_overall_prec = teach_overall_prec+teach_avg_prec
        teach_overall_rec = teach_overall_rec+teach_avg_rec

    '''Valores promedio de recall y precision para Fmeasure'''
    teach_avg_prec = teach_overall_prec/len(teach_docs)
    teach_avg_rec = teach_overall_rec/len(teach_docs)
    write_file("Teacher's average precision: " + str(teach_avg_prec) +
               "\nTeacher's average recall: " + str(teach_avg_rec))
    #our_avg_prec= our_avg_rec/len(our_docs)
    #our_avg_rec= our_avg_rec/len(our_docs)
    '''F1'''
    #our_F1 = f_beta(our_avg_prec,our_avg_rec)
    write_file("\nTeacher's F1 Score: " +
               str(f_beta(teach_avg_prec, teach_avg_rec, 1)))
    '''MAP'''
    write_file("\nTeacher's MAP: " + str(MAP(vTeachMAP, ref_docs)))
    '''p at n'''
    write_file("\nTeacher's precision at 5: " + str(teachAt5/len(teach_docs)))
    '''r-Precision'''
    write_file("\nTeacher's R-precision: " + str(teachRPrec/len(teach_docs)))
    '''MRR'''
    write_file("\nTeacher's MRR: " + str(teachMRR/len(teach_docs)))
    '''MRE'''

    #our_rprec = ourRPrec/len(teach_docs)


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
