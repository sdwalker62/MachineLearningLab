import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from drain3.drain import Drain
import numpy as np

collections = joblib.load("results/collections.joblib")
labels = joblib.load("results/labels.joblib")
containers = joblib.load("results/containers.joblib")
cd = joblib.load("results/matrices_dict.joblib")
dd = joblib.load("results/drain_dict.joblib")

def find_max_value(matrix_dict: dict)->int:
    max_value = 0
    for _, d1 in matrix_dict.items(): #collections
        for _, d2 in d1.items(): #labels
            for _, d3 in d2.items(): #containers
                test_value = np.amax(d3)
                if test_value > max_value:
                    max_value = test_value

    return max_value



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Label('Collection'),
    dcc.Dropdown(
        id='collections',
        options=[{"label": idx, "value": idx} for idx in collections],
        value='1',
        multi=True
    ),

    html.Label('Label'),
    dcc.Dropdown(
        id='labels',
        options=[{"label": l, "value": l} for l in labels],
        value='healthy',
        multi=True
    ),

    html.Label('Container'),
    dcc.Dropdown(
        id='containers',
        options=[{"label": c, "value": c} for c in containers],
        value='core.soaesb',
        multi=True
    ),

    dcc.Graph(id='heatmap')
], style={'columnCount': 1})


@app.callback(
    Output('heatmap', 'figure'),
    Input('collections', 'value'),
    Input('labels', 'value'),
    Input('containers', 'value'))
def update_heatmap(collections_set, labels_set, containers_set):
    # rows will always be containers and columns can either be labels or collections
    if not (len(collections_set) > 1 & len(labels_set) > 1):
        n_cols = len(collections_set) if len(collections_set) > 1 else len(labels_set)
        mdict = {i: {} for i in range(len(containers_set))}
        cdict = {i: {} for i in range(len(containers_set))}
        for i in range(len(containers_set)):
            for j in range(n_cols):
                if len(collections_set) > 1:
                    mdict[i][j] = cd[j][labels][containers_set[i]]
                else:
                    if len(labels_set)>1:
                        mdict[i][j] = cd[int(collections_set)][labels_set[j]][containers_set[i]]
                        cdict[i] = [cluster.get_template() for cluster in dd[containers_set[i]].clusters]
                    else:
                        mdict[i][j] = cd[int(collections_set)][labels_set][containers_set[i]]
        n_cols = len(collections_set) if len(collections_set) > 1 else len(labels_set)

        fig = make_subplots(
            rows = len(containers_set),
            cols = n_cols,
            start_cell = "top-left"
        )

        fig.update_yaxes(showticklabels=False)
#        fig.update_layout(margin=dict(t=100, r=100, b=100, l=100),
#                          width=2000, height=1200,
#                          autosize=False)
        fig.update_coloraxes(
            cmin = 0,
            cmax = find_max_value(cd)
        )

        for i in range(len(containers_set)):
            for j in range(n_cols):
                fig.add_trace(
                    go.Heatmap(z=mdict[i][j].tolist(),
                               y=cdict[i]),
                    row=i+1,
                    col=j+1)

    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
