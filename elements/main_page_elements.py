from dash import Input, Output, dcc, html, callback_context
import dash_bootstrap_components as dbc

navbar = dbc.NavbarSimple(
    children = [
        html.Img(
            src='./assets/logo.png',
            style={
                'height': '5vh',
                'align': 'left'}
        )],
        brand = '',
        brand_href="#",
        color="dark",
        style = {
            'border':'0px solid transparent'
        }
    )

drop_div = [html.Div(
        ["Drag and drop or click to select the MRI to analyse."],
        style = {
            "textAlign":   "center",
            "line-height": "20px"
        }
    )]

success_div = [html.Div(
        ["Your file was successfully uploaded."],
        style = {
            "textAlign":   "center",
            "line-height": "20px"
        }
    )]

upload = html.Div(
    dcc.Loading(
        id="loading-1",
        children=[
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    drop_div
                ),
                style={
                    "width": "100%",
                    "min-height": "20px",
                    },
                multiple=False
            )
        ],
    ),
    style = {
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px"
    }
)

predict = html.Div(
    [
        dbc.Button("Predict",  id = 'predict-button', color="primary")
    ],
    className="d-grid gap-2",
    style={'padding-top':'20px'}
)

reset = html.Div(
    [
        #dbc.Button("Restart",  id = 'reset-button', color="danger")
    ],
    className="d-grid gap-2",
    style={'padding-top':'20px'}
)

result = html.Div(
    [
        html.P('Press the predict button to evaluate the MRI.',
                style = {"align": "center"},
                id='result-div')
    ],
    className="result-div",
    style={'padding-top':'40px',
            'padding-bottom': '10px',
            "textAlign": "center"}
)