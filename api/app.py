r'''
'''
# third party imports
from flask import Flask, jsonify, request
from infer import Infer

app = Flask(__name__)


def get_inference(inp, model_name):
    infer = Infer(model_name)
    inference = infer.infer(inp=inp)
    return inference


@app.route('/predict', methods=['POST'])
def infer():
    if request.method == 'POST':
        # first, get the numeric from user input
        number = request.files['file']
        class_id, class_name = get_inference(inp=number)
        return jsonify({'class_id': class_id, 'class_name': class_name})