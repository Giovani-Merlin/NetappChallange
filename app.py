import dash
import numpy as np
from dash import dcc, callback_context
import dash_bootstrap_components as dbc
from elements import *

data = np.load('assets/mri.npy')

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    #suppress_callback_exceptions=True
    )

app.title = "KneeAI | UsGuri Team"

plotly_fig = plot_img(data,0)

app.layout = dbc.Container([
    navbar,
    dbc.Row(
            [
                dbc.Col(
                    # left-hand side column
                    children = [
                        dcc.Markdown(open('./assets/body.md').read())
                    ],
                    width=4,
                    style={"height": "100%", 
                            "padding": "15px", 
                            "align": "center"}
                ),
                dbc.Col(
                    # right-hand side column
                    children = [
                        dcc.Graph(id= 'mri-graph', figure=plotly_fig),
                        dcc.Slider(
                            id='slider',
                            min=0,
                            max=data.shape[0]-1,
                            step=1,
                            value=0),
                        result
                    ],
                    width = 8
                )
            ],
            style={"min-height": "60vh"}
        ),
    dbc.Row(
        [
            dbc.Col(
                reset,
                width = 4
                ),
            dbc.Col(
                predict,
                width = 8
                )
        ]
    )
    ],
    style={"height": "100vh"}
    )

@app.callback(
    dash.dependencies.Output('mri-graph', 'figure'),
    [dash.dependencies.Input('slider', 'value')]
)
def update_figure(slider):
    #create some graph   
    fig = plot_img(data,slider)
    return fig

@app.callback(
    dash.dependencies.Output('result-div', 'children'),
    [dash.dependencies.Input('predict-button', 'n_clicks')]
)
def update_result(n_clicks):
    #create some graph   
    changed_id = [p['prop_id'] for p in callback_context.triggered]
    
    if 'predict-button.n_clicks' in changed_id:
        return 'Abnormal injury detect | Probability: 76% | Threshold: 20%'
    else:
        return 'Press the predict button to evaluate the MRI.'

if __name__ == '__main__':
    app.run_server(debug=True)