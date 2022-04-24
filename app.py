import dash
import numpy as np
from dash import dcc, callback_context
import dash_bootstrap_components as dbc
from elements import *
import torch
import argparse
import numpy as np
from eval import eval_image
import os

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

def format_result(answer, prob, threshold):
    if answer:
        return f'Abnormal injury detect | Probability: {round(prob[0],3)} | Threshold: {threshold}'
    else:
        return f'Abnormal injury not detect | Probability: {round(prob[0],3)} | Threshold: {threshold}'

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
    
    image_path = 'assets/mri.npy'

    base_folder = os.getenv("BASE_FOLDER",".")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, 
                        choices=['abnormal', 'acl', 'meniscus'], default="abnormal")
    parser.add_argument('--plane', type=str,
                        choices=['sagittal', 'coronal', 'axial'], default='axial')
    parser.add_argument('--model_path', type=str, 
                        default="./models/model.pth")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--test_path", type=str, default=f"{base_folder}/data/")
    parser.add_argument('--image_path', type=str, help="As a numpy matrix")
    parser.add_argument('--threshold', type=int, default=0.2)
    args = parser.parse_args()
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    threshold=0.5

    answer, prob = eval_image(image_path, model, threshold=threshold)

    if 'predict-button.n_clicks' in changed_id:
        return format_result(answer, prob, threshold)
    else:
        return format_result(answer, prob, threshold)

if __name__ == '__main__':
    app.run_server(debug=True)