import os
import numpy as np
#import dash
#import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from flask import Flask, send_from_directory
import json
from urllib.parse import quote as urlquote
import cv2

import base64
import io
from imageio import imread

import requests


UPLOAD_DIRECTORY = "../uploaded" # <--TODO
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = Dash(server=server)

app.title = 'Video action recognition (ver. 0.1)'
app.layout = html.Div(
                    [

                    html.Div([
                                # dcc.Store(id='annotation_store', data=[]),

                                html.H2('Распознавание действий на видео'),
                                dcc.Upload(
                                            id='upload_video',
                                            children=html.Div([
                                                html.A('Загрузить видеофайл')
                                            ]),
                                            style={
                                                'width': '50%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'color': '#b904e5',
                                                'backgroundColor': '#b904e5',
                                                'opacity': 1
                                            },
                                            style_active = {"borderColor": '#03fe33', "backgroundColor": '#b904e5', 'opacity': 0.8},
                                            style_reject = {"borderColor": '#f50057', "backgroundColor": '#b904e5', 'opacity': 0.8},

                                            multiple=False
                                        ),
                                html.H3("Загруженный файл"),
                                html.Div(id='file_list', children="-"),
                                html.Iframe(src='http://{}:8000/'.format('127.0.0.1'), # <--TODO
                                            style={'width': '50%', 'height': '500px'}),
                                html.H3("Распознанные классы"),
                                html.Div(id='detected_classes', children="-"),

                            ], style={'padding-top': '0%', 'padding-left': '0%'})
                    ]
                )


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


@app.callback(
    Output("file_list", "children"),
    Output("detected_classes", "children"),
    [Input("upload_video", "filename"), Input("upload_video", "contents")],
)
def update_output(uploaded_filename, uploaded_file_content):

    if uploaded_filename is not None and uploaded_file_content is not None:
        save_file(uploaded_filename, uploaded_file_content)
        url = "http://127.0.0.1:8000/get?video=../uploaded/{}".format(uploaded_filename)# <--TODO
        response = requests.get(url=url)
        clss = '{}'.format(response.content.decode())
        return uploaded_filename, clss
    return 'Нет видео', '-'


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)# <--TODO
