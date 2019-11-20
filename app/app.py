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
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# internal imports
# from infer import Infer       # TODO: is this neccesary?

app = Flask(__name__)


model = Infer(model_name=None)


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