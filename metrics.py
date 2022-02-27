import json
import argparse
import numpy as np
import pandas as pd
import plotly as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import math
from plotly.subplots import make_subplots


def load_docs(arg):
    path = '.\corpora'
    path2 = '\json\qrels.json'
    corpus = '\\'+arg.c
    '''REF'''
    with open(path+corpus+path2) as f:
        ref_docs = json.loads(f.read())
    '''OURS'''
    with open(path+corpus+"\json\cf_results.json") as f:
        our_docs = json.loads(f.read())
    '''TEACHER'''
    with open('.\TFIDF_reference_results'+corpus+'_ref_qresults.json') as f:
        teach_docs = json.loads(f.read())
    '''PUNT TEACHER'''
    with open(path+corpus+'\json\qscoredrels.json') as f:
        punt_docs_teach = json.loads(f.read())
    '''PUNT OURS'''
    with open(path+corpus+'\json\qscoredrels.json') as f:
        punt_docs_ours = json.loads(f.read())
    return pd.json_normalize(ref_docs), pd.json_normalize(teach_docs), pd.json_normalize(our_docs), pd.json_normalize(punt_docs_teach), pd.json_normalize(punt_docs_ours)


def get_args():
    parser = argparse.ArgumentParser(description='Paso de parámetros')
    parser.add_argument("-c", dest="c")  # corpus
    parser.add_argument("-rf", dest="rf")  # result-file path
    return parser.parse_args()


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


def f_beta(avgPrec, avgRec, beta):
    if avgPrec == 0 or avgRec == 0:
        return 0
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


'''Multilevel Relevance Evaluation , utilizaremos n documentos'''


def MRE(relDocs, compDocs, puntDocs, nDocs):
    CG = []
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

    return CG, nDocs


def avg_prec_rec(relDocs, compDocs):
    hits = 0
    if len(compDocs) == 0:
        return 0, 0
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
        if n == 0:
            return 0
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
        if n == 0:
            return 0
    for i in range(n):
        if compDocs[i] in relDocs:
            hits += 1
    prec = hits/n
    return prec


def metrics(ref_docs, teach_docs, our_docs, punt_docs_teach, punt_docs_our):
    vTeachPrec = []
    vOurPrec = []
    vTeachMAP = []
    vOurMAP = []
    vTeachMRE = []
    vOurMRE = []
    vTeachRPrec = []
    vOurRPrec = []
    F1MacroTeach = 0
    F1MacroOur = 0
    teach_overall_prec = 0
    teach_overall_rec = 0
    our_overall_prec = 0
    our_overall_rec = 0
    teachAt5 = 0
    ourAt5 = 0
    teachAt10 = 0
    ourAt10 = 0
    teachRPrec = 0
    ourRPrec = 0
    teachMRR = 0
    ourMRR = 0
    for index in ref_docs.index:
        # lista con los documentos relevantes para una query
        relDocs = ref_docs["relevantDocs"][index]
        teachDocs = teach_docs["relevantDocs"][index]
        ourDocs = our_docs["relevantDocs"][index]
        puntDocsTeach = punt_docs_teach["relevantDocs"][index]
        puntDocsOur = punt_docs_our["relevantDocs"][index]

        '''Precision-Recall Normalizado'''
        teachPrec = prec_rec(relDocs, teachDocs)
        vTeachPrec.append(teachPrec)
        ourPrec = prec_rec(relDocs, ourDocs)
        vOurPrec.append(ourPrec)
        '''MAP - Mean Average Precision'''
        ourMAP = MAP_vector(relDocs, ourDocs)
        vOurMAP.append(ourMAP)
        teachMAP = MAP_vector(relDocs, teachDocs)
        vTeachMAP.append(teachMAP)
        '''p at N'''
        teachAt5 = teachAt5 + p_at_n(relDocs, teachDocs, 5)
        ourAt5 = ourAt5 + p_at_n(relDocs, ourDocs, 5)
        teachAt10 = teachAt10 + p_at_n(relDocs, teachDocs, 10)
        ourAt10 = ourAt10 + p_at_n(relDocs, ourDocs, 10)
        '''r-Precision'''
        tRPrec = rPrecision(relDocs, teachDocs)
        oRPrec = rPrecision(relDocs, ourDocs)
        vTeachRPrec.append(tRPrec)
        vOurRPrec.append(oRPrec)
        teachRPrec = teachRPrec + tRPrec
        ourRPrec = ourRPrec + oRPrec
        '''MRR'''
        teachMRR = teachMRR + MRR(relDocs, teachDocs)
        ourMRR = ourMRR + MRR(relDocs, ourDocs)
        '''MRE'''
        #CG,n = MRE(relDocs, teachDocs, puntDocsTeach,15)
        # vTeachMRE.append(CG)

        '''Valores totales de recall y precision para Fmeasure'''
        our_avg_prec, our_avg_rec = avg_prec_rec(relDocs, ourDocs)
        F1MacroOur = F1MacroOur + (f_beta(our_avg_prec, our_avg_rec, 1))
        our_overall_prec = our_overall_prec+our_avg_prec
        our_overall_rec = our_overall_rec+our_avg_rec
        teach_avg_prec, teach_avg_rec = avg_prec_rec(relDocs, teachDocs)
        F1MacroTeach = F1MacroTeach + \
            (f_beta(teach_avg_prec, teach_avg_rec, 1))
        teach_overall_prec = teach_overall_prec+teach_avg_prec
        teach_overall_rec = teach_overall_rec+teach_avg_rec
    '''Valores promedio de recall y precision para Fmeasure'''
    teach_avg_prec = teach_overall_prec/len(teach_docs)
    teach_avg_rec = teach_overall_rec/len(teach_docs)
    our_avg_prec = our_overall_prec/len(our_docs)
    our_avg_rec = our_overall_rec/len(our_docs)
    F1MacroTeach = F1MacroTeach/len(our_docs)
    F1MacroOur = F1MacroOur/len(our_docs)
    write_file("Teacher's average precision: " + "{:.3f}".format(teach_avg_prec) +
               "\nOur average precision: " + "{:.3f}".format(our_avg_prec) +
               "\nTeacher's average recall: " + "{:.3f}".format(teach_avg_rec) +
               "\nOur average recall: " + "{:.3f}".format(our_avg_rec))
    '''F1'''
    write_file("\nTeacher's F1 Micro Score: " +
               "{:.3f}".format(f_beta(teach_avg_prec, teach_avg_rec, 1))
               + "\nOur F1 Micro Score: " +
               "{:.3f}".format(f_beta(our_avg_prec, our_avg_rec, 1)))
    write_file("\nTeacher's F1 Macro Score: " +
               "{:.3f}".format(F1MacroTeach)
               + "\nOur F1 Macro Score: " +
               "{:.3f}".format(F1MacroOur))
    '''MAP'''
    write_file("\nTeacher's MAP: " + "{:.3f}".format(MAP(vTeachMAP, ref_docs)) +
               "\nOur MAP: " + "{:.3f}".format(MAP(vOurMAP, ref_docs)))
    '''p at n'''
    write_file("\nTeacher's precision at 5: " + "{:.3f}".format(teachAt5/len(teach_docs)) +
               "\nOur precision at 5: " + "{:.3f}".format(ourAt5/len(our_docs)))
    write_file("\nTeacher's precision at 10: " + "{:.3f}".format(teachAt10/len(teach_docs)) +
               "\nOur precision at 10: " + "{:.3f}".format(ourAt10/len(our_docs)))
    '''r-Precision'''
    write_file("\nTeacher's R-precision: " + "{:.3f}".format(teachRPrec/len(teach_docs)) +
               "\nOur R-precision: " + "{:.3f}".format(ourRPrec/len(our_docs)))
    '''MRR'''
    write_file("\nTeacher's MRR: " + "{:.3f}".format(teachMRR/len(teach_docs))
               + "\nOur MRR: " + "{:.3f}".format(ourMRR/len(our_docs)))
    '''MRE'''

    '''Compare metrics and graphs'''
    # prec_rec_curve(vOurPrec,vTeachPrec)
    # queries_bars(vOurRPrec,vTeachRPrec)
    # CG_DCG_curve()
    #nine_Bars(ourAt10, our_docs, teachAt10, teach_docs, teachAt5, ourAt5, our_avg_prec, our_avg_rec, teach_avg_prec, teach_avg_rec, ourRPrec, teachRPrec, F1MacroOur, F1MacroTeach, ourMRR, teachMRR, vOurMAP, ref_docs, vTeachMAP)


def nine_Bars(ourAt10, our_docs, teachAt10, teach_docs, teachAt5, ourAt5, our_avg_prec, our_avg_rec, teach_avg_prec, teach_avg_rec, ourRPrec, teachRPrec, F1MacroOur, F1MacroTeach, ourMRR, teachMRR, vOurMAP, ref_docs, vTeachMAP):

    fig = make_subplots(rows=3, cols=3,
                        subplot_titles=("Precision", "Recall", "F1 Macro", "F1 Micro",
                                        "R-Precision", "MRR", "MAP", "P at 5", "P at 10")
                        )
    l = [(ourAt10/len(our_docs)), teachAt10/len(teach_docs)]
    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  3, 3)
    fig.update_xaxes(dtick=1)
    l = [(ourAt5/len(our_docs)), teachAt5/len(teach_docs)]
    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  3, 2)
    l = [f_beta(our_avg_prec, our_avg_rec, 1),
         f_beta(teach_avg_prec, teach_avg_rec, 1)]
    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  2, 1)
    l = [ourRPrec/len(our_docs), teachRPrec/len(teach_docs)]
    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  2, 2)
    l = [F1MacroOur, F1MacroTeach]
    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  1, 3)
    l = [ourMRR/len(our_docs), teachMRR/len(teach_docs)]

    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  2, 3)
    l = [MAP(vOurMAP, ref_docs), MAP(vTeachMAP, ref_docs)]

    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  3, 1)
    l = [our_avg_prec, teach_avg_prec]

    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  1, 1)
    l = [our_avg_rec, teach_avg_rec]

    fig.add_trace(go.Bar(x=["Nuestro", "Profesor"], y=l,
                         marker=dict(color=l, coloraxis="coloraxis")),
                  1, 2)

    fig.update_layout(coloraxis=dict(colorscale='twilight'), showlegend=False)
    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="black"
    )

    fig.write_html('images/9bars.html',
                   auto_open=True)
    fig.write_image("images/9bars.png")


def CG_DCG_curve(teach, our):
    length = len(our)
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=list(range(0, length)), y=our,
                                   name="Our", line_color='black', mode='lines+markers',))
    fig = fig.add_trace(go.Scatter(x=list(range(0, length)), y=teach,
                                   connectgaps=True,
                                   name='Teacher', line_color='red', mode='lines+markers'))
    fig = fig.update_layout(xaxis_title='Query',
                            yaxis_title='NDCG', title='NDCG Vector')
    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="black"
    )

    fig.write_html('images/NDCG.html',
                   auto_open=True)
    fig.write_image("images/NDCG.png")


def queries_bars(vOurRPrec, vTeachRPrec):
    '''Gráfico de barras con la RPrec para cada query'''
    df = pd.DataFrame()
    df['RPrecision'] = vOurRPrec
    df['num'] = range(0, len(vOurRPrec))

    fig = px.bar(df, x="num", y='RPrecision',
                 title="Our R-Precision", color="RPrecision")
    fig = fig.update_layout(xaxis_title='Query',
                            yaxis_title='RPrecision')

    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="black"
    )

    fig.update_xaxes(dtick=5)
    fig.write_html('images/rprec_bars.html',
                   auto_open=True)
    fig.write_image("images/rprec_bars.png")

    df['RPrecision'] = np.subtract(vOurRPrec, vTeachRPrec)
    df['num'] = range(0, len(vOurRPrec))

    fig = px.bar(df, x="num", y='RPrecision',
                 title="Our R-Precision - Teacher R-Precision", color="RPrecision")
    fig = fig.update_layout(xaxis_title='Query',
                            yaxis_title='RPrecision')

    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="black"
    )

    fig.update_xaxes(dtick=5)
    fig.write_html('images/rprec_bars_comp.html',
                   auto_open=True)
    fig.write_image("images/rprec_bars_comp.png")


def average_curve(vector):
    ret_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(0, 11):
        for i in range(len(vector)):
            ret_vector[j] = ret_vector[j] + vector[i][j]
        ret_vector[j] = float("{:.3f}".format(ret_vector[j]/len(vector)))
    return ret_vector


def prec_rec_curve(ourVector, teacherVector):
    ourVector = average_curve(ourVector)
    teacherVector = average_curve(teacherVector)
    recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig = go.Figure()
    fig = fig.update_yaxes(range=[0, 1])
    fig = fig.add_trace(go.Scatter(x=recall, y=ourVector,
                        name="Our", line_color='black'))
    fig = fig.add_trace(go.Scatter(x=recall, y=teacherVector,
                                   connectgaps=True,
                                   name='Teacher', line_color='red', mode='lines+markers'))
    fig = fig.update_layout(xaxis_title='Recall',
                            yaxis_title='Precision', title='Precision vs Recall')
    fig.update_layout(
        font_color="black",
        title_font_color="red",
        legend_title_font_color="black"
    )

    fig.write_html('images/prec_curves.html',
                   auto_open=True)
    fig.write_image("images/prec_curves.png")


def main():
    f = open("metrics.txt", "w")
    f.write("")
    f.close()
    args = get_args()
    ref_docs, teach_docs, our_docs, punt_docs_teach, punt_docs_ours = load_docs(
        args)
    metrics(ref_docs, teach_docs, our_docs, punt_docs_teach, punt_docs_ours)


if __name__ == '__main__':
    if not os.path.exists("images"):
        os.mkdir("images")
    main()
