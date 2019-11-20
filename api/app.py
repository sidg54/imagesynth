r'''
'''
# third party imports
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# internal imports
from infer import Infer

app = Flask(__name__)