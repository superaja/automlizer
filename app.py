import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import datetime
import time
from util import hyperProfiler, hyperprofilercharts, pipeProfile, dataLoader, \
    topnPipeline, hyperTable, algoOptions, kpi, percentfig, algoHist, fileLoader, \
    Semaphore, preprocess, realTimeStatusFigs, percentStatus, create_card, algoHistDetails
from aml import runTPOT, createsklearnPipeline, pipelineRef
from os import path
import os
from math import log
import time

# constants
kpiStyle = {"text-align": "center", "font-size": "20px", "font-family": "Roboto, sans-serif", "color": "orange"}
divStyle = {"background": "black", "height": "50px", "padding": "10px 0"}

tableStyle = {"font-size": "15px", "font-family": "verdana, sans-serif"}
dragDropStyle={'width': '100%', 'height': '20px', 'lineHeight': '20px', 'borderWidth': '1px', \
                'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '2px'}
cmetrics = {'label': 'Accuracy', 'value': 'accuracy'}
rmetrics = {'label': 'Root Mean Squared Error', 'value': 'Root Mean Squared Error'}
n=10

semaphore = Semaphore()
global stat
stat = {"gen": 0, "pipeline": [], "score":[], "pct_algo": 0, "pct_hyper":0, "algo_hist":[], \
         "time_start": 0, "time_end": 0, "status": "", "highest_score": 0,  \
         "evalgo": 0, "evalpipes": 0, "evalhyper": 0, 'top_n_pipe': [], "top_n_score": [], "nAlgoHist": {} }

app = dash.Dash(__name__, meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])
app.title = "AutoMLizer"

def long_process(gen, X, y, metric, algo):
    global stat
    score = []
    pipeline_name = []
    if semaphore.is_locked():
        raise Exception('Resource is locked')
    total_pipes = 0

    for i in range(gen):
        semaphore.lock()
        stat['status'] = 'Started'
        po, po_score, pipes = runTPOT(X,y, metric, algo)
        createsklearnPipeline(po, total_pipes)
        total_pipes = total_pipes + pipes
        df_pipe, df_ref = dataLoader('pipeline.csv', 'ref.csv')
        pct_algo, pct_hyper = percentStatus(df_pipe, df_ref)
        highest_score, evalgo, evalpipes, evalhyper = kpi(df_pipe, algo)
        df_pipeline_score = df_pipe[['PIPELINE','SCORE']].copy().drop_duplicates()
        #df_pipeline_score = df_pipeline_score[df_pipeline_score['SCORE']>0.1]
        pipeline_name = pipeline_name + list(df_pipeline_score['PIPELINE'])
        score = score + list(df_pipeline_score['SCORE'])
        print (len(score))
        # top 10 pipeline
        df_top_10 = df_pipeline_score[['PIPELINE', 'SCORE']].copy().drop_duplicates()
        if algo == 'Classifier':
            df_top_10 = df_top_10.sort_values(['SCORE'], ascending=False).nlargest(n, ['SCORE'])
        else:
            df_top_10 = df_top_10.sort_values(['SCORE'], ascending=False).nsmallest(n, ['SCORE']) 
        # Update real-time dict
        stat['top_n_pipe'] = ["P-"+ str(e) for e in list(df_top_10['PIPELINE'])]
        stat['top_n_score'] = list(df_top_10['SCORE'])
        stat['pipeline'] = pipeline_name
        stat['score'] = score
        stat['highest_score'] = highest_score
        stat['evalgo'] = evalgo
        stat['evalpipes'] = evalpipes
        stat['evalhyper'] = evalhyper
        stat['pct_algo'] = pct_algo
        stat['pct_hyper'] = pct_hyper
        stat['gen'] = i+1
        stat['time_start'] = datetime.datetime.now()
        semaphore.unlock()
        stat['time_end'] = datetime.datetime.now()
        #time.sleep(2)
    stat['status'] = 'Completed'
    return (stat['time_end'] - stat['time_start']).total_seconds()*1e6, stat


pfig1, pfig2 = realTimeStatusFigs()

# App Layout

app.layout = html.Div([
                html.Div(id="output-clientside"),
                
                html.Div([
                    html.H3("AutoMLyzer", style=kpiStyle, className='mx-auto'),
                ], className='row border-bottom', style=divStyle), # Header Row
                
                html.Div([
                    html.Div([ # Left Section
                        html.Div([
                            html.Div([
                                html.Div(["Run Specifications"],className='card-header'),
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            dcc.Upload(
                                            id='upload-file',
                                            children=html.Div([html.Img(src=app.get_asset_url('upload.png'), height=30),
                                            ], className='col-3 align-self-center')),
                                            html.H6("File Name: ", className='col-4 align-self-center'),
                                            html.H6(id='uploaded-file', className='col-4'),
                                            ], className='row col-12'),
                                        html.Br(),
                                        html.Div([
                                            ], className='row'),
                                    ], className='row'),
                                    html.Br(),
                                    html.Div([
                                        html.H6("Label: "),
                                        html.Div([
                                            dcc.Dropdown(
                                                id = 'label-dropdown',
                                                options = [{'label': 'Empty Results', 'value': 'NA'}],
                                                        )
                                        ], className='dropdown col-8'),
                                    ], className='row'),
                                    html.Br(),
                                    html.Div([
                                        html.Div([
                                            dcc.RadioItems(
                                            id = 'algo-select',
                                            options=[
                                                {'label': 'Classifier  ', 'value': 'Classifier'},
                                                {'label': 'Regressor  ', 'value': 'Regressor'},
                                            ],
                                            value='Classifier',
                                            labelStyle={'display': 'inline-block', 'cursor': 'pointer', "margin-right": "10px"})
                                            ], className='col-8'),
                                        ], className='row'),
                                    html.Br(),
                                    html.Div([
                                    html.H6("Metric: "),
                                    html.Div([
                                    dcc.Dropdown(
                                        id='metric-dropdown',
                                        options = [cmetrics, rmetrics]
                                                )
                                    ], className='dropdown col-8'),
                                    ], className='row'),
                                    html.Br(),
                                    html.Div([
                                    html.H6("Budget (Generations): "),
                                    ], className='row'),                                 
                                    html.Div([
                                        dcc.Slider(
                                            id = 'budget-slider',
                                            min=10,
                                            max=500,
                                            #step=30,
                                            marks={i:"{}".format(i) for i in range(50,500, 50)},
                                            value=5
                                        )  
                                    ],),
                                    html.Br(),
                                    html.Br(),
                                    html.Div([
                                        html.Div([
                                        html.Button("Submit", className='btn btn-secondary', id='button', n_clicks=0),
                                        ], className = 'col-12 text-center'),
                                    ], className='row'),           
                                ], className='card-body')
                            ], className='card col-12')
                        ], className='row'),
                    
                    html.Br(),
                    html.Div([
                    html.Div(id='status'),
                    html.Div(id='time'),
                    ], className='row'),
                    html.Div(id='lock'),
                    html.Br(),
                    # Realtime Status
                    html.Div([
                        pfig1,
                        pfig2,
                        html.Div([
                            daq.Tank(
                                id='gen-tank',
                                showCurrentValue=True,
                                units = 'Gens',
                                min=0)  
                        ], className='col-4'),
                        dcc.Interval(
                            id='interval-component',
                            interval=2*1000, # in milliseconds
                            n_intervals=0
                            ),
                    ], className='row'),
                    
                    # Algo Hist

                    html.Div([
                        create_card('', dcc.Graph(
                            id = 'algo_hist',
                            config={
                                'displayModeBar': False
                            }
                        )),
                    ], className='row'),

                    ], className='col-3'), # left Section

                    html.Div([], className='ml-2'),

                    html.Div([ # Right Section
                        html.Div([
                    
                    html.Div([
                        create_card('Pipelines Evaluated', html.H5(id='evalpipes'))
                    ], className='col-3'),

                    html.Div([
                        create_card('Algorithms Evaluated', html.H5(id='evalgo'))
                    ], className='col-3'),

                    html.Div([
                        create_card('Hyper Evaluated', html.H5(id='evalhyper'))
                        ], className='col-3'),
                    html.Div([
                        create_card('Best Score', html.H5(id='highest_score'))
                        ], className='col-3'),
                    dcc.Interval(
                            id='kpi-interval-component',
                            interval=2*1000, # in milliseconds
                            n_intervals=0
                            ),
                ], className='row'), # KPI Row

                html.Div(html.Br(), className='row', style={"height": "10px"}),

                html.Div([
                    
                    html.Div([
                        create_card('',
                        dcc.Graph(
                            id='pipeline_profile',
                            config={
                                'displayModeBar': False
                            }
                        )),
                        dcc.Interval(
                            id='pipeline-interval-component',
                            interval=2*1000, # in milliseconds
                            n_intervals=0
                            )
                    ], className='col-12'),
                ], className='row'), # Pipeline Profile Section

                # Algorithm Histograms
                html.Div(html.Br(), className='row', style={"height": "10px"}), # spacer

                html.Div([
                        html.Div(id='algo_hist_details', className='col-12')
                    ], className='row'),
                
                html.Div([   
                    html.Div([
                        create_card('',
                        dcc.Graph(
                            id='selected_pipe',
                            config={
                                'displayModeBar': False
                            }
                            ))
                        ], className='col-6'),
                        
                    html.Div(id='pipeline_info', className='col-3')

                ], className='row'), # Top 10 Pipeline Detail Sections

                
                html.Div([
                    html.Br(),
                    html.Div([dcc.RadioItems(
                                id='radio_items', 
                                options=[{"label": "Select Pipeline above", "value":"Select Pipeline above"}],
                                labelStyle={'display': 'inline-block'},
                                className='col-12 mt-1')
                            ])
                ], className='row'), # Div for Drop Down

                html.Div([
                    html.Br(),
                    html.Div([
                        html.Div(id='drop-graph', className='col-12')
                    ], className='row'),
                    html.Br(),
                    html.Div([
                        html.Div(id='drop-graph1', className='col-12')
                    ], className='row')

                ], className='row') # Div for Hypers
                        
            ], className='col-8'), # Right Section

            ], className='row'),

            html.Div([
                html.H6("Obsidian Steel Copyright © 2019–2020", style=kpiStyle, className='mx-auto')
            ], className='row', style=divStyle) # This is the footer

            ], className='container-fluid', style={'backgroundColor':'white'}) # main container


# call backs

# TO DO: ADD A CALLBACK FOR RESETTING THE APP TO ORIGINAL VALUES FOR NEW RUN

@app.callback(
    [Output('label-dropdown', 'options'),
    Output('uploaded-file', 'children')],
    [Input('upload-file', 'filename')]
)
def file_upload(filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        options = [{"label": "Empty File", "value": "Empty File"}]
        fName = "Empty"
    else:
        options, fName = fileLoader(filename)
    return options, fName

@app.callback(
    Output('gen-tank', 'max'),
    [Input('budget-slider', 'value')]
)
def updategenMax(geninput):
    return geninput

@app.callback(
    [Output('lock', 'children'),
    Output('time', 'children'),
    Output('interval-component', 'disabled'),
    Output('kpi-interval-component', 'disabled'),
    Output('pipeline-interval-component', 'disabled')],
    [Input('button', 'n_clicks')],
    [#State('gen-input', 'value'),
    State('algo-select', 'value'),
    State('metric-dropdown', 'value'),
    State('budget-slider', 'value'),
    State('uploaded-file', 'children'),
    State('label-dropdown', 'value')
    ]
    )
def run_process(n_click, algo, metric, gen, filename, label):
    # receive gen, filename, label, budget
    if n_click > 0:
        # remove previous files
        if path.exists('pipeline.csv'):
            os.remove('pipeline.csv')
        if path.exists('ref.csv'):
            os.remove('ref.csv')
        p_ref = pipelineRef(algo)
        X, y =  preprocess(filename, label, algo)
        total_time, stat = long_process(gen, X, y, metric, algo)
        stat['status'] = 'Completed'
        print("waiting for 3 seconds to finish processes")
        time.sleep(3)
        return ' ', ' : ' + str(total_time)+ " Seconds", True, True, True
    else: 
        return ' ', ' ', False, False, False 

@app.callback([
    Output('status', 'children'),
    Output('button', 'children'),
    Output('gen-tank', 'value'),
    Output('algo-status-fig', 'value'),
    Output('hyper-status-fig', 'value')],
    [Input('interval-component', 'n_intervals')])
def display_status(n):
    return stat['status'], 'Submit' if stat["status"] in ['Completed', ''] else 'Running', \
        stat['gen'], stat['pct_algo'], stat['pct_hyper']


@app.callback([
    Output('evalpipes', 'children'),
    Output('evalgo', 'children'),
    Output('evalhyper', 'children'),
    Output('highest_score', 'children')],
    [Input('kpi-interval-component', 'n_intervals')]
)
def display_kpi_status(n):
    return stat['evalpipes'], stat['evalgo'], stat['evalhyper'], stat['highest_score']

@app.callback(
    Output('pipeline_profile', 'figure'),
    [Input('pipeline-interval-component', 'n_intervals')]
)
def display_pipeline_profile(n):
    figure={
            'data':[ 
                go.Scatter(
                mode = 'markers',
                x = stat['pipeline'],
                y = stat['score'],
                marker = {"symbol":"circle", "size": 6, "color": "orange"},
            )],
            'layout': go.Layout(
                title='Pipeline Performance Profile',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )}
    return figure

@app.callback(
    Output('selected_pipe', 'figure'),
    [Input('pipeline-interval-component', 'n_intervals')]
)
def display_top_n_pipelines(n):
    topnfig = {
                'data': [
                    go.Bar(
                        y = stat['top_n_pipe'],
                        x = stat['top_n_score'],
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

@app.callback(
    Output('algo_hist', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def display_algo_hist(n):
    fig = {
    'data': [
        go.Histogram(
        y = stat['score'],
        marker = dict(color='orange')
    )
    ],
    'layout':
        go.Layout(
        title = 'Pipeline Histogram',
        width = 400,
        height = 400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.1, 
        )
        
    }
    return fig


@app.callback([
    Output('drop-graph', 'children'),
    Output('drop-graph1', 'children')
    ],
    [Input('radio_items', 'value')]
)
def update_graph(selected_algo):
    df_pipe, df_ref = dataLoader('pipeline.csv', 'ref.csv')
    algo_df = df_pipe.copy()
    algo = selected_algo
    d1, d2 = hyperProfiler(algo, algo_df)
    plnc, plnn = hyperprofilercharts(d1,d2, algo)
    num_temp_divs = []
    num_final_divs = []
    cat_temp_divs = []
    cat_final_divs = []
    for i in plnn: # create subsets of 2 graphs
        num_temp_divs.append(i)
        if len(num_temp_divs) == 2:
            num_final_divs.append(html.Div([html.Br(), html.Div(num_temp_divs, className='row')]))
            num_temp_divs=[]
    for i in plnc: # create subsets of 2 graphs
        cat_temp_divs.append(i)
        if len(cat_temp_divs) == 2:
            cat_final_divs.append(html.Div([html.Br(),html.Div(cat_temp_divs, className='row')]))
            cat_temp_divs=[]
    return cat_final_divs, num_final_divs

@app.callback(
    [Output('radio_items', 'options'),
    Output('radio_items', 'value'),
    Output('pipeline_info', 'children')],
    [Input('selected_pipe', 'selectedData')]
)
def update_pipeline(selectedData):
    df_pipe, df_ref = dataLoader('pipeline.csv', 'ref.csv')
    algo_df = df_pipe.copy()
    df_pipeline_table = df_pipe[['PIPELINE', 'ALGO_NAME', 'HYPER_NAME', 'HYPER_VALUE']]
    if selectedData is None:
        raise PreventUpdate
    # get the values from df and return it
    if selectedData is not None: 
        selectedPipeline = selectedData['points'][0]['y']
        selectedPipeline = selectedPipeline[2:]
        pipe = int(float(selectedPipeline))
        radio_o, radio_v = algoOptions(df_pipeline_table[df_pipeline_table['PIPELINE']==pipe])
        return radio_o, radio_v, hyperTable(df_pipeline_table[df_pipeline_table['PIPELINE']==pipe])



@app.callback(
    Output('algo_hist_details', 'children'),
    [Input('status', 'children')]
)
def algo_hist_detail_display(status):
    num_temp_divs = []
    num_final_divs = []
    if status == 'Completed':
        df_pipe, df_ref = dataLoader('pipeline.csv', 'ref.csv')
        df_algo = df_pipe[df_pipe['PP_FLAG']=='N']
        algo_hist_detail_div = algoHistDetails(df_algo)
        return html.Div(algo_hist_detail_div, className='row')
    else:
        return dcc.Loading(type='graph', children=html.Div([html.H6("Generating Individual Algorithm Histograms....")]))

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)