import pandas as pd
import numpy as np
import plotly.graph_objs as go
import random
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from math import log
import dash_daq as daq
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# algo hist

def algoHistDetails(d1, algo_name=None):
    algoHistFig = []
    for algo in list(d1.ALGO_NAME.unique()):
        d1 = d1[['ALGO_NAME', 'PP_FLAG', 'SCORE']].copy().drop_duplicates()
        algo_score_df = d1[d1['ALGO_NAME']==algo]
        algo_score = list(algo_score_df['SCORE'])
        # histogram
        fig = {
            'data': [
                go.Histogram(
                y = algo_score,
                marker = dict(color='orange')
            )
            ],
            'layout':
                go.Layout(
                title = algo,
                width = 300,
                height = 300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0.2
                )
                
            }
        algoHistFig.append(
            create_histcard('',
            html.Div([
                dcc.Graph(
                id=str(round(random.randint(1, 100))) + algo + ' Hist',
                figure = fig,
                config={'displayModeBar': False},
                )
            ])) #
        )
    return algoHistFig


# hyper charts
def hyperprofilercharts(d1, d2, algo_name=None):
    # for type=C
    plc=[]
    for hv in list(d2.HYPER_NAME.unique()):
        hyper_name_df = d2[d2['HYPER_NAME']==hv]
        for v in hyper_name_df['HYPER_VALUE'].unique():
            if v is not None:
                hyper_profiler_df = hyper_name_df[hyper_name_df['HYPER_VALUE']==v]
                pipeline = ["P-" + str(pipe) for num, pipe in enumerate(list(hyper_profiler_df.PIPELINE))]
                if pipeline[0] =='7.0': pipeline[0]='7.0_Stack'
                score = list(hyper_profiler_df['SCORE'])
                p = {
                    'data': [
                        go.Bar(
                            x = pipeline,
                            y = score,
                            orientation='v',
                            marker=dict(color='orange')
                        )
                    ],
                    'layout': go.Layout(
                        title= hv + '=' + v,
                        showlegend=False,
                        #width=400,
                        height=200,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis = {'title': 'pipeline'},
                        yaxis = {'title': "score", "showticklabels": False}, 
                        
                        )
                    }

                plc.append(
                    create_minicard('', 
                    html.Div([
                        dcc.Graph(
                        id=v + ' ' + str(round(random.randint(1, 100))) + algo_name + ' Hist',
                        figure = p,
                        config={
                                    'displayModeBar': False
                                },
                    )
                    ])) #
                    )
    
    # for type=N
    pln=[]
    for hv in list(d1.HYPER_NAME.unique()):
        hyper_name_df = d1[d1['HYPER_NAME']==hv]
        #print(hyper_name_df)
        pipeline = ["P=" + str(pipe) for num, pipe in enumerate(list(hyper_name_df.PIPELINE))]
        if pipeline[0] =='7.0': pipeline[0]='7.0_Stack'
        hypervalue=list(hyper_name_df['HYPER_VALUE'])
        if pd.isnull(hypervalue).any() == False:
            score = list(hyper_name_df['SCORE'])
            p = {
                'data': [
                    go.Scatter(
                        x = hypervalue,
                        y = score,
                        mode='markers',
                        marker=dict(color='orange')
                    )
                ],
                'layout': go.Layout(
                    title=  hv,
                    showlegend=False,
                    #width=300,
                    height=300,
                    xaxis = {'title': hv},
                    yaxis = {'title': "score"},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                    )
                }
            pln.append(
                    create_minicard('',
                    html.Div([
                        dcc.Graph(
                        id= hv + ' ' + str(round(random.randint(1, 100))) + algo_name  + ' Hist',
                        figure=p,
                        config={
                                'displayModeBar': False
                            }
                        )
                    ])
                    ) #
                    )
    return plc, pln
 
# df filter for model
def hyperProfiler(modelName, algo_df):
    numericHyperdf = algo_df[(algo_df['ALGO_NAME']==modelName) & (algo_df['HYPER_TYPE']=='N')]
    stringHyperdf = algo_df[(algo_df['ALGO_NAME']==modelName) & (algo_df['HYPER_TYPE']=='C')]
    return numericHyperdf, stringHyperdf
 
# data loader
 
def dataLoader(fName, rName):
    df_pipe = pd.read_csv(fName)
    df_ref = pd.read_csv(rName)
    # filter the preprocessors
    df_pipe = df_pipe[df_pipe['PP_FLAG']=='N']
    return df_pipe, df_ref

def fileLoader(fName):
    options = []
    if not fName: 
        options = [{'label': "Upload File", 'value': "Upload File"}]
        fName = ""
    else: 
        df = pd.read_csv(fName)
        cols = df.columns
        for c in cols:
            options.append({'label': c, 'value': c})
    return options, fName

# Algorithm Histogram

def algoHist(df_algo):

    fig = {
    'data': [
        go.Histogram(
        y = df_algo['SCORE'],
        marker = dict(color='orange')
    )
    ],
    'layout':
        go.Layout(
        title = 'Algorithm Histogram',
        showlegend=False,
        width = 400,
        height = 400,
        bargap=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
        
    }
    return fig


# pipeline profile
''' 
def pipeProfile(df_pipeline_score):
    marker={"symbol":"circle", "size": 6, "color": "orange"}
    fig1 =  {
    'data':[ 
        go.Scatter(
        x = df_pipeline_score['PIPELINE'],
        y = df_pipeline_score['SCORE'],
        mode = 'markers',
        marker = marker,
    )],
    'layout': go.Layout(
        title='Pipeline Performance Profile',
        #width = 1000,
        #height = 500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )}
    
    return fig1

'''
def pipeProfile():
    marker={"symbol":"circle", "size": 6, "color": "orange"}
    fig1 =  {
    'data':[ 
        go.Scatter(
        mode = 'markers',
        marker = marker,
    )],
    'layout': go.Layout(
        title='Pipeline Performance Profile',
        #width = 1000,
        height = 350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )}
    
    return fig1


def topnPipeline(df_pipeline_score, n):
    df_top_10 = df_pipeline_score[['PIPELINE', 'SCORE']].copy().drop_duplicates()
    df_top_10 = df_top_10.sort_values(['SCORE'], ascending=False).nlargest(n, ['SCORE'])
 
    topnfig = {
                'data': [
                    go.Bar(
                        y = ["P-"+ str(e) for e in list(df_top_10['PIPELINE'])],
                        x = list(df_top_10['SCORE']),
                        #x1 = [abs(log(e)) for e in df_top_10['SCORE']],
                        #marker=go.bar.Marker(color='rgb(55, 83, 108)'),
                        marker=go.bar.Marker(color='orange'),
                        orientation='h'
                    )
                ],
                'layout': go.Layout(
                    title='Top 10 Pipelines',
                    showlegend=False,
                    legend=go.layout.Legend(
                        x=1.0,
                        y=0,
                    ),
                    #width=500,
                    height=465,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    clickmode='event+select' 
                )
                
            }
    return topnfig

def percentfig(pipe_df, ref_df):
    # Number of Evaluated Algorithms
    #evalgo = len(pipe_df['ALGO_NAME'].unique())
    evalgo = len(pipe_df['ALGO_NAME'][pipe_df['ALGO_NAME'].str.contains('_stack')==False][pipe_df['PP_FLAG']=='N'].unique())
    # Hyperparameters
    evalhyper = len(pipe_df['HYPER_NAME'].unique())
    # Total Algo Space
    total_algo = len(ref_df['ALGO_NAME'].unique())
    print(total_algo)
    # Total Hyper Space
    total_hyper = ref_df['TOTAL_HYPER_COUNT'].groupby(ref_df['ALGO_NAME']).count().sum()

    pct_algos = (evalgo / total_algo)*100
    pct_hypers = (evalhyper / total_hyper)*100

    pfig1 = html.Div([
                daq.Gauge(
                    id='my-gauge',
                    color={"gradient":True,"ranges":{"green":[60,100],"yellow":[40,59],"red":[0,39]}},
                    label="% Algorithms",
                    size=150,
                    value=round(pct_algos, 0),
                    min=0,
                    max=100)
            ])
    
    pfig2 = html.Div([
                daq.Gauge(
                    id='my-gauge-1',
                    color={"gradient":True,"ranges":{"green":[0,6],"yellow":[6,8],"red":[8,10]}},
                    label="% Hyperparameters",
                    size = 150,
                    value=round(pct_hypers, 0),
                    min=0,
                    max=100)
            ]) 
    
    return pfig1, pfig2




def hyperTable(dataframe, max_rows=10):
    return html.Table(
    # Header
    [html.Tr([html.Th(col) for col in dataframe.columns])] +

    # Body
    [html.Tr([
        html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
    ]) for i in range(min(len(dataframe), max_rows))]
,className='col-3 table', style={"font-size": "12px"})

def algoOptions(dataframe):
    # get options 
    algo_filter = list(dataframe['ALGO_NAME'].unique())
    print (algo_filter)
    algo_dropdown_filter = []
    for algo in algo_filter:
        option_dict = {'label': algo, 'value': algo}
        algo_dropdown_filter.append(option_dict)
    return algo_dropdown_filter, algo_dropdown_filter[0]['value']


def kpi(df, algo):
    # Highest Score
    if algo == 'Classifier':
        highest_score = df[['PIPELINE', 'SCORE']].nlargest(1, 'SCORE')
    else:
        highest_score = df[['PIPELINE', 'SCORE']].nsmallest(1, 'SCORE') 
    print (highest_score)
    # Number of Evaluated Algorithms
    #evalgo = len(df['ALGO_NAME'].unique())
    evalgo = len(df['ALGO_NAME'][df['ALGO_NAME'].str.contains('_stack')==False][df['PP_FLAG']=='N'].unique())
    # Number of Pipelines
    evalpipes = len(df['PIPELINE'].unique())
    # Hyperparameters
    evalhyper = len(df['HYPER_NAME'].unique())

    return round(highest_score['SCORE'].values[0], 6), evalgo, evalpipes, evalhyper

class Semaphore:
    def __init__(self, filename='semaphore.txt'):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write('done')

    def lock(self):
        with open(self.filename, 'w') as f:
            f.write('working')

    def unlock(self):
        with open(self.filename, 'w') as f:
            f.write('done')

    def is_locked(self):
        return open(self.filename, 'r').read() == 'working'


def preprocess(filename, label, algo):
    df = pd.read_csv(filename)
    ldf = df[label]
    tdf = df[[c for c in df.columns if c not in [label]]]
    if algo =='Classifier':
        le = LabelEncoder()
        ldf = pd.DataFrame(le.fit_transform(ldf), columns=[label])
    else:
        pass
    X = tdf.values.astype(np.float)
    y = ldf.values.astype(np.float)
    return X, y

def realTimeStatusFigs():
    algo_status_fig = html.Div([
        daq.Gauge(
            id='algo-status-fig',
            label="% Algorithms",
            size=130,
            min=0,
            max=100)
    ], className='col-4')

    hyper_status_fig = html.Div([
        daq.Gauge(
            id='hyper-status-fig',
            label="% Hyperparameters",
            size=130,
            min=0,
            max=100)
    ], className='col-4')

    return algo_status_fig, hyper_status_fig

def percentStatus(pipe_df, ref_df):
    # Number of Evaluated Algorithms
    #evalgo = len(pipe_df['ALGO_NAME'].unique())
    evalgo = len(pipe_df['ALGO_NAME'][pipe_df['ALGO_NAME'].str.contains('_stack')==False][pipe_df['PP_FLAG']=='N'].unique())
    # Hyperparameters
    evalhyper = len(pipe_df['HYPER_NAME'].unique())
    # Total Algo Space
    total_algo = len(ref_df['ALGO_NAME'].unique())
    print(evalgo, evalhyper)
    # Total Hyper Space
    total_hyper = ref_df['TOTAL_HYPER_COUNT'].groupby(ref_df['ALGO_NAME']).count().sum()

    pct_algos = (evalgo / total_algo)*100
    pct_hypers = (evalhyper / total_hyper)*100

    return round(pct_algos, 0), round(pct_hypers, 0)


def create_card(title, graph):

    card = html.Div([
            html.Div([
                html.Div([
                    html.H5(title, className='card-title'),
                    html.Div(graph),
                ])

            ], className='card-body')

    ], className='card col-12')

    return card

def create_minicard(title, graph):

    card = html.Div([
            html.Div([
                html.Div([
                    html.H5(title, className='card-title'),
                    html.Div(graph),
                ])

            ], className='card-body')

    ], className='card col-6')

    return card

def create_histcard(title, graph):

    card = html.Div([
            html.Div([
                html.Div([
                    html.H5(title, className='card-title'),
                    html.Div(graph),
                ])

            ], className='card-body')

    ], className='card col-3')

    return card