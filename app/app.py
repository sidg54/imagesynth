r'''
Main app file to expose flask APIs
for inference.
'''
# standard library imports
import os

# third party imports
import pickle
import pandas as pd
import torch
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


def load_model(model_name):
    '''
    Loads a model of the given name.

    Arguments
    ---------
        model_name : str
            Name of the model to load.
    
    Returns
    -------
        obj
            Model object loaded from a .pt/.pth file.
    '''
    filepath = f'{os.getcwd()}/trained_models/{model_name}'
    if not (filepath[-3:] == '.pt' or filepath[-4:] == '.pth'):
        filepath = filepath + '.pt'

    model = torch.load(filepath)
    return model


# flask code below here
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/imagesynth/gan')
def generate_image():
    if request.json:
        data = request.json
        input_path = data.input_path
        print(input_path)
        r = requests.get(input_path, timeout=60)
        # save input to disk
        temp_file = 'temp/temp.txt'
        f = open(temp_file, 'wb')
        f.write(r.content)
        f.close()

        