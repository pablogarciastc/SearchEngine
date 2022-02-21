import json
import argparse
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import matplotlib.pyplot as plt
import numpy as np

def funcion1(propios, relevantes):
    precision = []
    recall = []
    hits = 0
    precision_norm = []
    recall_norm = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for j in range(len(propios)):
        if propios[j] in relevantes:
            hits += 1
            prec = hits / (j + 1)
            rec = hits / len(relevantes)
            precision.append(prec)
            recall.append(rec)
    # normalizacion a 11 niveles
    if not precision:
        precision_norm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        contador = 0
        for k in range(recall_norm.__len__()):
            if contador != (recall.__len__() - 1):
                if k == 0:
                    precision_norm.append(np.amax(precision[contador:]))
                else:
                    if recall_norm[k] > recall[contador]:
                        # max(precision[contador:])
                        contador += 1
                        precision_norm.append(np.amax(precision[contador:]))
                    else:
                        precision_norm.append(np.amax(precision[contador:]))
            else:
                if recall[contador] >= recall_norm[k]:
                    precision_norm.append(np.amax(precision[contador:]))
                else:
                    precision_norm.append(0)
    return precision_norm

def funcion2(vectores, tamaño):
    vector = []
    for i in range(vectores.__len__()):
        if i == 0:
            for j in range(vectores[i].__len__()):
                vector.append(vectores[i][j])
        else:
            for j in range(vectores[i].__len__()):
                vector[j] = vector[j] + vectores[i][j]

    for j in range(vector.__len__()):
        vector[j] = vector[j] / tamaño

    return vector

def funcion3(propios, relevantes):
    hits = 0
    for j in range(len(propios)):
        if propios[j] in relevantes:
            hits += 1
    if not propios:
        return [0, 0]
    elif not relevantes:
        return [0, 0]
    else:
        prec = hits / len(propios)
        rec = hits / len(relevantes)
        return [prec, rec]

def mAP(array_propios, array_relevantes):
    array_precision = []
    for i in range(array_propios.__len__()):
        relevantes = array_relevantes[i]['relevantDocs']
        propios = array_propios[i]['relevantDocs']

        precision = []
        hits = 0
        for j in range(len(propios)):
            if propios[j] in relevantes:
                hits += 1
                prec = hits / (j + 1)
                precision.append(prec)
        array_precision.append(precision)

    AP_total = 0
    for l in range(array_precision.__len__()):
        ap = 0
        if array_precision[l]:
            for k in range(array_precision[l].__len__()):
                if k == 0:
                    ap = ap + array_precision[l][k]
                else:
                    if array_precision[l][k - 1] == array_precision[l][k]:
                        continue
                    else:
                        ap = ap + array_precision[l][k]
            ap = ap / array_precision[l].__len__()

        AP_total = AP_total + ap

    AP_total = AP_total / array_propios.__len__()
    return AP_total

def funcion5(propios, relevantes, N):
    hits = 0
    if not propios:
        return 0

    if propios.__len__() < N:
        min = propios.__len__()
    else:
        min = N

    for j in range(min):
        if propios[j] in relevantes:
            hits += 1
    hits = hits / N
    return hits

def rPrecision(array_propios, array_relevantes):
    r_precision = []

    for i in range(array_propios.__len__()):
        relevantes = array_relevantes[i]['relevantDocs']
        propios = array_propios[i]['relevantDocs']

        R = relevantes.__len__()
        hits = 0
        prec = 0

        if (propios.__len__() < R):
            min = propios.__len__()
        else:
            min = R

        for j in range(min):
            if propios[j] in relevantes:
                hits += 1
        prec = hits / (R)

        r_precision.append(prec)

    return r_precision

def mRR(array_propios, array_relevantes):
    suma_position = 0
    threshold = 10

    for i in range(array_propios.__len__()):
        relevantes = array_relevantes[i]['relevantDocs']
        propios = array_propios[i]['relevantDocs']

        position = 1000

        for j in range(propios.__len__()):
            if propios[j] in relevantes:
                position = j + 1
                break

        if (position > threshold):
            RR = 0
        else:
            RR = (1/position)

        suma_position += RR

    mRR = suma_position/array_propios.__len__()

    return mRR


def multilevel(array_propios, array_relevantes, array_puntuaciones ):
    multilevel = []

    for i in range(array_propios.__len__()):

        relevantes = array_relevantes[i]['relevantDocs']
        propios = array_propios[i]['relevantDocs']
        puntuaciones = array_puntuaciones[i]['relevantDocs']
        multilevel_query_i = []

        longitud = 25
        if (propios.__len__() < longitud): #Numero de resultados con los que vamos a trabajar
            min = propios.__len__()
        else:
            min = longitud

        for j in range(min):

            if propios[j] in relevantes:

                indice = relevantes.index(propios[j])
                puntuacion = puntuaciones[indice]['relevance']
                puntuacion_suma = (int(puntuacion[0]) + int(puntuacion[1]) + int(puntuacion[2]) + int(puntuacion[3])) / 4

                if j == 0:
                    puntuacion_suma_log = puntuacion_suma
                else:
                    puntuacion_suma_log = puntuacion_suma / math.log2(j+1)

                if multilevel_query_i:
                    multilevel_query_i.append(multilevel_query_i[-1]+puntuacion_suma_log)
                else:
                    multilevel_query_i.append(puntuacion_suma_log)
            else:
                if multilevel_query_i:
                    multilevel_query_i.append(multilevel_query_i[-1] )
                else:

                    multilevel_query_i.append(0)

        multilevel.append(multilevel_query_i)
        for p in range(longitud - multilevel[0].__len__()):
            multilevel[0].append(0)

        multilevel_media = np.array(multilevel[0])

    for s in range(1,multilevel.__len__()):

        for p in range(longitud -multilevel[s].__len__()):
            multilevel[s].append(0)

        multilevel_media = multilevel_media + np.array(multilevel[s])

    multilevel_media = multilevel_media/ multilevel.__len__()

    return multilevel_media



# Obtener parámetros de entrada
parser = argparse.ArgumentParser(description='Paso de parámetros')
parser.add_argument("-c", dest="c", help="corpus sobre el que hacer las métricas")
parser.add_argument("-rf", dest="rf", help="archivo para almacenar los resultados")
params = parser.parse_args()

# Cargar documentos relevantes de nuestro searcher
file_our_docs = open("results_" + params.c + ".json", 'r')
our_docs = file_our_docs.read()
our_docs_json = json.loads(our_docs)

# Cargar documentos relevantes de Santos
file_juan_docs = open("TFIDF_reference_results/" + params.c + "_ref_qresults.json", 'r')
juan_docs = file_juan_docs.read()
juan_docs_json = json.loads(juan_docs)

# Cargar documentos relevantes de referencia
file_rels_docs = open("corpora/" + params.c + "/json/qrels.json", 'r')
rels_docs = file_rels_docs.read()
rels_docs_json = json.loads(rels_docs)

# Inicio métricas
# Precision vs recall, MAP variables
vectores_nuestros = []
vectores_santos = []
recall_norm = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#F-Measure variables
precision1_final = 0
recall1_final = 0
precision2_final = 0
recall2_final = 0
beta = 1
# Precision at N
N = 5
precision5_nuestros = 0
precision5_santos = 0
precision10_nuestros = 0
precision10_santos = 0

for i in range(rels_docs_json.__len__()):
    relevantes = rels_docs_json[i]['relevantDocs']
    nuestros = our_docs_json[i]['relevantDocs']
    santos = juan_docs_json[i]['relevantDocs']

    #Precision vs recall, MAP
    precision_norm = funcion1(nuestros, relevantes)
    vectores_nuestros.append(precision_norm)
    precision2_norm = funcion1(santos, relevantes)
    vectores_santos.append(precision2_norm)

    #F-Measure
    valores_nuestros = funcion3(nuestros, relevantes)
    valores_santos = funcion3(santos, relevantes)
    precision1_final = precision1_final + valores_nuestros[0]
    recall1_final = recall1_final + valores_nuestros[1]
    precision2_final = precision2_final + valores_santos[0]
    recall2_final = recall2_final + valores_santos[1]

    #Precision at N
    valor5_nuestro = funcion5(nuestros, relevantes, 5)
    precision5_nuestros = precision5_nuestros + valor5_nuestro
    valor5_santos = funcion5(santos, relevantes, 5)
    precision5_santos = precision5_santos + valor5_santos
    valor10_nuestro = funcion5(nuestros, relevantes, 10)
    precision10_nuestros = precision10_nuestros + valor10_nuestro
    valor10_santos = funcion5(santos, relevantes, 10)
    precision10_santos = precision10_santos + valor10_santos

#Precision vs recall result
vector_nuestro = funcion2(vectores_nuestros, rels_docs_json.__len__())
vector_santos = funcion2(vectores_santos, rels_docs_json.__len__())
fig = plt.figure("Recall vs Precision")
plt.ylim(0.0, 1.05)
plt.xlim(0.0, 1.05)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.plot(recall_norm, vector_nuestro, marker='o')
plt.plot(recall_norm, vector_santos, marker='o')
plt.draw()

#Precision y recall
precision1_final = precision1_final / rels_docs_json.__len__()
recall1_final = recall1_final / rels_docs_json.__len__()
precision2_final = precision2_final / rels_docs_json.__len__()
recall2_final = recall2_final / rels_docs_json.__len__()
print("Nuestra precision es de: " + str(precision1_final))
print("La precision de Santos es: " + str(precision2_final))
print("Nuestro recall es de: " + str(recall1_final))
print("El recall de Santos es: " + str(recall2_final))

#Precision at 5 y 10
precision5_nuestros = precision5_nuestros / rels_docs_json.__len__()
precision5_santos = precision5_santos / rels_docs_json.__len__()
precision10_nuestros = precision10_nuestros / rels_docs_json.__len__()
precision10_santos = precision10_santos / rels_docs_json.__len__()
print("Nuestra precision @ 5 es de: " + str(precision5_nuestros))
print("La precision @ 5 de Santos es: " + str(precision5_santos))
print("Nuestra precision @ 10 es de: " + str(precision10_nuestros))
print("La precision @ 10 de Santos es: " + str(precision10_santos))

#F-Measure
Fmeasure_nuestro = (beta**2 + 1)*((precision1_final*recall1_final)/((precision1_final*(beta**2))+recall1_final))
Fmeasure_santos = (beta**2 + 1)*((precision2_final*recall2_final)/((precision2_final*(beta**2)+recall2_final)))
print("Nuestro F-Measure es: " + str(Fmeasure_nuestro))
print("El F-Measure de Santos es: " + str(Fmeasure_santos))

#MAP result
AP_nuestro = mAP(our_docs_json,rels_docs_json)
AP_santos = mAP(juan_docs_json,rels_docs_json)
print("Nuestro MAP es de: " + str(AP_nuestro))
print("El MAP de Santos es de: " + str(AP_santos))

#RPrecision
r_precision_nuestro = rPrecision(our_docs_json,rels_docs_json)
r_precision_santos = rPrecision(juan_docs_json,rels_docs_json)
np_r_precision_nuestro = np.array(r_precision_nuestro)
np_r_precision_santos= np.array(r_precision_santos)

r_precision_comparativa = np_r_precision_nuestro - np_r_precision_santos

x = np.arange(1,101,1)
fig = plt.figure("R-Precision")
plt.bar(x,r_precision_comparativa)
plt.xlabel('Query Number')
plt.ylabel('R-Precision Nuestro/Referencia')
plt.grid(True)
plt.draw()

#mRR
mrr_nuestro = mRR(our_docs_json,rels_docs_json)
mrr_santos = mRR(juan_docs_json,rels_docs_json)
print("Nuestro mRR es de: " + mrr_nuestro.__str__())
print("El mRR de Santos es de: " + mrr_santos.__str__())

#Impresion F-Measure y MAP
fig = plt.figure("Métricas")
subplot_precision = fig.add_subplot(221)
subplot_recall = fig.add_subplot(222)
subplot_Fmeasure = fig.add_subplot(223)
subplot_map = fig.add_subplot(224)

nombres = ['Nuestro','Santos']
datos_precision = [precision1_final,precision2_final]
datos_recall = [recall1_final, recall2_final]
datos_Fmeasure = [Fmeasure_nuestro,Fmeasure_santos]
datos_map = [AP_nuestro, AP_santos]

xx_precision = range(len(datos_precision))
xx_recall = range(len(datos_recall))
xx_Fmeasure = range(len(datos_Fmeasure))
xx_map = range(len(datos_map))

subplot_precision.bar(xx_precision, datos_precision, align='center')
subplot_precision.set_xticks(xx_precision)
subplot_precision.set_xticklabels(nombres)
subplot_precision.set_ylabel("Precision")
subplot_precision.set_ylim([0.0, 1.0])

subplot_recall.bar(xx_recall, datos_recall, align='center')
subplot_recall.set_xticks(xx_recall)
subplot_recall.set_xticklabels(nombres)
subplot_recall.set_ylabel("Recall")
subplot_recall.set_ylim([0.0, 1.0])

subplot_Fmeasure.bar(xx_Fmeasure, datos_Fmeasure, align='center')
subplot_Fmeasure.set_xticks(xx_Fmeasure)
subplot_Fmeasure.set_xticklabels(nombres)
subplot_Fmeasure.set_ylabel("F-Measure B=1")
subplot_Fmeasure.set_ylim([0.0, 1.0])

subplot_map.bar(xx_map, datos_map, align='center')
subplot_map.set_xticks(xx_map)
subplot_map.set_xticklabels(nombres)
subplot_map.set_ylabel("MAP")
subplot_map.set_ylim([0.0, 1.0])
plt.draw()


#multilevel

if (params.c == "cf"):
    # Cargar documentos relevantes de referencia con puntuacion
    file_rels_puntuacion_docs = open("corpora/" + params.c + "/json/qscoredrels.json", 'r')
    rels_puntuacion_docs = file_rels_puntuacion_docs.read()
    rels_puntuacion_docs_json = json.loads(rels_puntuacion_docs)

    multilevel_nuestro = multilevel(our_docs_json,rels_docs_json,rels_puntuacion_docs_json)
    multilevel_santos = multilevel(juan_docs_json,rels_docs_json,rels_puntuacion_docs_json)

    x = np.arange(1,26,1)
    fig = plt.figure("DCG")
    plt.xlabel('Rank')
    plt.ylabel('AVG(DCG)')
    plt.grid(True)
    plt.plot(x, multilevel_nuestro, marker='o')
    plt.plot(x, multilevel_santos, marker='o')
    plt.draw()

plt.show()
print("FIN")
