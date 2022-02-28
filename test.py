from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


vTeachMRE = []

def MRE(relDocs, compDocs, puntDocs, nDocs):
    CG = []
    aux = nDocs
    if len(compDocs) < nDocs:
        nDocs = len(compDocs)

    for i in range(nDocs):
        if compDocs[i] in relDocs:
            # se coge los 4 valores de relevancia de esa coincidencia
            relevance = puntDocs[relDocs.index(compDocs[i])]['relevance']
            avg_relevance = (
                int(relevance[0]) + int(relevance[1]) + int(relevance[2]) + int(relevance[3])) / 4
            if i == 0:
                avg_relevance = avg_relevance
            else:
                avg_relevance = avg_relevance / math.log2(i+1)

            if CG:
                CG.append(avg_relevance+CG[-1])
            else:
                CG.append(avg_relevance)

        else:
            if CG:
                CG.append(CG[-1])
            else:
                CG.append(0)
    if(nDocs<aux):
        for i in range(0,aux-nDocs):
            CG.append(CG[-1])

    return CG, nDocs


puntDocsTeach = [{'relevantDoc': 268, 'relevance': '2222'}, {'relevantDoc': 324, 'relevance': '2222'}, {'relevantDoc': 449, 'relevance': '2122'}, {'relevantDoc': 992, 'relevance': '1220'}, {'relevantDoc': 1191, 'relevance': '1001'}]
[59, 183, 370, 579, 803, 833, 1000, 1017, 1033, 1097, 1232]
[{'relevantDoc': 59, 'relevance': '0101'}, {'relevantDoc': 183, 'relevance': '2222'}, {'relevantDoc': 370, 'relevance': '1211'}, {'relevantDoc': 579, 'relevance': '2222'}, {'relevantDoc': 803, 'relevance': '1100'}, {'relevantDoc': 833, 'relevance': '0110'}, {'relevantDoc': 1000, 'relevance': '1212'}, {'relevantDoc': 1017, 'relevance': '2212'}, {'relevantDoc': 1033, 'relevance': '0001'}, {'relevantDoc': 1097, 'relevance': '1100'}, {'relevantDoc': 1232, 
'relevance': '0101'}]

DCGteach,n = MRE(relDocs, teachDocs, puntDocsTeach,25)
vTeachMRE.append(DCGteach)

vMREavgTeach= [sum(i) for i in zip(*vTeachMRE)]
vMREavgTeach = [vMREavgTeach / len(vTeachMRE) for vMREavgTeach in vMREavgTeach]

