from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def normalize_precision(precision, recall):  # normalizar a Standard 11-level
    normRec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    normPrec = []
    aux = 0
    if not precision:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

def normalize_precision2(precision, recall):  # normalizar a Standard 11-level
    normRec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    normPrec = []
    precNorm = [0,0,0,0,0,0,0,0,0,0,0]
    aux = 0
    p = np.array(precision)
    m = np.zeros(p.size, dtype=bool)
    excptIndx=[]
    for i in range(len(precision)): 
        m[excptIndx] = True
        a = np.ma.array(p, mask=m)
        precNorm[i]=precision[np.argmax(a)]
        excptIndx.append(i)
    print(precNorm)

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
    print(precision)
    precision = normalize_precision2(precision, recall)
    return precision

relDocs1 = [3,56,129]
relDocs2 = [3,5,9,25,39,44,56,71,89,123]
Docs = [123,84,56,6,8,9,511,129,187,25,38,48,250,113,3]
Docs1Prec = prec_rec(relDocs1,Docs)
Docs2Prec = prec_rec(relDocs2,Docs)

print(Docs2Prec)

  