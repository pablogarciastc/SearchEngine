from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
our = [0.86, 0.74, 0.68, 0.63, 0.58, 0.56, 0.53, 0.5, 0.47, 0.02, 0.01,0.86, 0.74, 0.68, 0.63, 0.58, 0.56, 0.53, 0.5, 0.47, 0.02, 0.01,0.86, 0.74, 0.68, 0.63, 0.58, 0.56, 0.53, 0.5, 0.47, 0.02, 0.01]
teach = [0.61, 0.52, 0.46, 0.4, 0.34, 0.3, 0.27, 0.24, 0.22, 0.2, 0.19,0.61, 0.52, 0.46, 0.4, 0.34, 0.3, 0.27, 0.24, 0.22, 0.2, 0.19,0.61, 0.52, 0.46, 0.4, 0.34, 0.3, 0.27, 0.24, 0.22, 0.2, 0.19]
length = len(our)
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=list(range(0,length)), y=our,
                        name="Our", line_color='black', mode='lines+markers',))
fig = fig.add_trace(go.Scatter(x=list(range(0,length)), y=teach,
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

