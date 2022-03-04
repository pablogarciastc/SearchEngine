'''
Pasos:
    1.Cambio de parámetros en bucle
    2.Computar F1 global para cada cambio de parámetros
    3.Almacenar resultado en json tal que:
        [
            {
                "F1": X,
                "K":X,
                "WEIGHT_...
            }, ...
        ]
    4.Buscar valor más alto de F1 y printear valores

'''

import variables
import searcher
import json
import numpy as np
import pandas as pd

def avg_prec_rec(relDocs, compDocs):
    hits = 0
    for i in range(len(compDocs)):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits/len(compDocs)
    rec = hits/len(relDocs)
    return prec, rec

def f_beta(avgPrec, avgRec, beta):
    return (beta**2+1)*(avgPrec*avgRec)/(beta**2*avgPrec+avgRec)

def F1(corpus,file):
    our_overall_prec = 0
    our_overall_rec = 0
    path = '.\corpora'
    path2 = '\json\qrels.json'
    # REF
    with open(path+corpus+path2) as f:
        ref_docs = json.loads(f.read())
    # OURS
    with open('.\maximizerResults'+file) as f:
       our_docs = json.loads(f.read())
    ref_docs=pd.json_normalize(ref_docs)
    our_docs=pd.json_normalize(our_docs)
    for index in ref_docs.index:
        relDocs = ref_docs["relevantDocs"][index]
        ourDocs = our_docs["relevantDocs"][index]
        '''Valores totales de recall y precision para Fmeasure'''
        our_avg_prec,our_avg_rec = avg_prec_rec(relDocs,ourDocs)
        our_overall_prec = our_overall_prec+our_avg_prec
        our_overall_rec = our_overall_rec+our_avg_rec

    '''Valores promedio de recall y precision para Fmeasure'''
    our_avg_prec= our_overall_prec/len(our_docs)
    our_avg_rec= our_overall_rec/len(our_docs)
    '''F1'''
    return f_beta(our_avg_prec,our_avg_rec,1)

def variandoWEIGHTcf(queries_json):
    file = open(".\F1Results\WEIGHTcf.txt", "w")
    for i in np.linspace(0.1,1,10): 
        variables.WEIGHT_ABSTRACT=i
        for j in np.linspace(i+0.2,1,8):
            if j<=1:
                variables.WEIGHT_MINOR=j
                for k in np.linspace(j+0.2,1,5):
                    if k<=1:
                        variables.WEIGHT_MAJOR=k
                        for p in np.linspace(k+0.2,1,3):
                            if p<=1:
                                variables.WEIGHT_TITLE=p
                                searcher.cf_queries_file(queries_json,".\maximizerResults\T"+
                                    str(p)+"MA"+str(k)+"MI"+str(j)+"A"+str(i)+".json")
                                file.write("T_"+str(p)+"__MA_"+str(k)+"__MI_"+str(j)+
                                    "__A_"+str(i)+"____#"+str(F1('\cf',"\T"+str(p)+"MA"+
                                    str(k)+"MI"+str(j)+"A"+str(i)+".json"))+"\n")
    file.close()

def variandoBCcf(queries_json):
    file = open(".\F1Results\BCcf.txt", "w")
    for i in np.linspace(0.1,1,10): 
        variables.BC_ABSTRACT=i
        for j in np.linspace(i+0.2,1,8):
            if j<=1:
                variables.BC_MINOR=j
                for k in np.linspace(j+0.2,1,5):
                    if k<=1:
                        variables.BC_MAJOR=k
                        for p in np.linspace(k+0.2,1,3):
                            if p<=1:
                                variables.BC_TITLE=p
                                searcher.cf_queries_file(queries_json,".\maximizerResults\BT"+
                                    str(p)+"BMA"+str(k)+"BMI"+str(j)+"BA"+str(i)+".json")
                                file.write("T_"+str(p)+"__MA_"+str(k)+"__MI_"+str(j)+
                                    "__A_"+str(i)+"____#"+str(F1('\cf',"\BT"+
                                    str(p)+"BMA"+str(k)+"BMI"+str(j)+"BA"+str(i)+".json"))+"\n")
    file.close()

def variandoKcf(queries_json):
    file = open(".\F1Results\Kcf.txt", "w")
    for p in np.linspace(1,5,50):
        variables.K=p
        searcher.cf_queries_file(queries_json,".\maximizerResults\K"+
            str(p)+".json")
        file.write("K_"+str(p)+"____#"+str(F1('\cf',"\K"+str(p)+".json"))+"\n")
    file.close()

def variandoWEIGHTmoocs(queries_json):
    file = open(".\F1Results\WEIGHTmoocs.txt", "w")
    for i in np.linspace(0.1,1,10): 
        variables.WEIGHT_DESCRIP=i
        for j in np.linspace(i+0.2,1,8):
            if j<=1:
                variables.WEIGHT_TITLEM=j
                searcher.moocs_queries_file(queries_json,".\maximizerResults\T"+
                    str(j)+"D"+str(i)+".json")
                file.write("T_"+str(j)+"__D_"+str(i)+"____#"+str(F1('\moocs',"\T"+str(j)+"D"+str(i)+".json"))+"\n")
    file.close()

def variandoBCmoocs(queries_json):
    file = open(".\F1Results\BCmoocs.txt", "w")
    for i in np.linspace(0.1,1,10): 
        variables.BC_DESCRIP=i
        for j in np.linspace(i+0.2,1,8):
            if j<=1:
                variables.BC_TITLEM=j
                searcher.moocs_queries_file(queries_json,".\maximizerResults\BT"+
                    str(j)+"BD"+str(i)+".json")
                file.write("T_"+str(j)+"__D_"+str(i)+"____#"+str(F1('\moocs',"\BT"+
                    str(j)+"BD"+str(i)+".json"))+"\n")
    file.close()

def variandoBCmoocs(queries_json):
    file = open(".\F1Results\BCmoocs.txt", "w")
    for i in np.linspace(0.1,1,10): 
        variables.BC_DESCRIP=i
        for j in np.linspace(i+0.2,1,8):
            if j<=1:
                variables.BC_TITLEM=j
                searcher.moocs_queries_file(queries_json,".\maximizerResults\BT"+
                    str(j)+"BD"+str(i)+".json")
                file.write("T_"+str(j)+"__D_"+str(i)+"____#"+str(F1('\moocs',"\BT"+
                    str(j)+"BD"+str(i)+".json"))+"\n")
    file.close()



def variandoKmoocs(queries_json):
    file = open(".\F1Results\Kmoocs.txt", "w")
    for p in np.linspace(1,5,10):
        variables.KM=p
        searcher.moocs_queries_file(queries_json,".\maximizerResults\K"+
            str(p)+".json")
        file.write("K_"+str(p)+"____#"+str(F1('\moocs',"\K"+str(p)+".json"))+"\n")
    file.close()

def variandoPredcf(queries_json):
    file = open(".\F1Results\Thresholdcf.txt", "w")
    for p in np.linspace(97.31,97.39,9):
        variables.CF=p
        searcher.cf_queries_file(queries_json,".\maximizerResults\Thresholdcf"+
            str(p)+".json")
        file.write("Umbral_"+str(p)+"____#"+str(F1('\cf',"\Thresholdcf"+str(p)+".json"))+"\n")
    file.close()

def variandoPredmoocs(queries_json):
    file = open(".\F1Results\Thresholdmoocs.txt", "w")
    for p in np.linspace(99.75,99.85,11):
        variables.MOOCS=p
        searcher.moocs_queries_file(queries_json,".\maximizerResults\Thresholdmoocs"+
            str(p)+".json")
        file.write("Umbral_"+str(p)+"____#"+str(F1('\moocs',"\Thresholdmoocs"+str(p)+".json"))+"\n")
    file.close()

def variandoUmbralcf(queries_json):
    file = open(".\F1Results\ThresholdFixedcf.txt", "w")
    for p in np.linspace(0.1,1,19):
        variables.CFUmbral=p
        searcher.cf_queries_file(queries_json,".\maximizerResults\ThresholdFixedcf"+
            str(p)+".json")
        file.write("Umbral_"+str(p)+"____#"+str(F1('\cf',"\ThresholdFixedcf"+str(p)+".json"))+"\n")
    file.close()

def variandoUmbralmoocs(queries_json):
    file = open(".\F1Results\ThresholdFixedmoocs.txt", "w")
    for p in np.linspace(0.1,1,19):
        variables.MOOCSUmbral=p
        searcher.moocs_queries_file(queries_json,".\maximizerResults\ThresholdFixedmoocs"+
            str(p)+".json")
        file.write("Umbral_"+str(p)+"____#"+str(F1('\moocs',"\ThresholdFixedmoocs"+str(p)+".json"))+"\n")
    file.close()

def main():
    ''' CF ''' 
    # with open('.\corpora\cf\json\queries.json') as f:
    #        queries_json = json.loads(f.read())
    
    # searcher.cf_queries_file(queries_json,".\maximizerResults\without.json")
    # file = open(".\F1Results\without.txt", "w")
    # file.write(str(F1('\cf',"\without.json"))+"\n")
    # file.close()
    
    #variandoWEIGHTcf(queries_json)
    #variandoBCcf(queries_json)
    #variandoKcf(queries_json)
    #variandoPredcf(queries_json)
    #variandoUmbralcf(queries_json)

    ''' MOOCS '''
    with open('.\corpora\moocs\json\queries.json') as f:
            queries_json = json.loads(f.read())
    
    #variandoWEIGHTmoocs(queries_json)
    #variandoBCmoocs(queries_json)
    #variandoKmoocs(queries_json)
    #variandoPredmoocs(queries_json)
    variandoUmbralmoocs(queries_json)
    
if __name__ == '__main__':
    main()