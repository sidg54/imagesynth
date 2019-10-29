r'''
Module to use already trained models for inference.
'''
# standard library imports
import os

# third party imports
import torch


class Infer:
    ''' Class to use already-trained models for inference. '''

    def __init__(self, model_name):
        '''
        Initializes an instance of the Infer class.

        Arguments
        ---------
            model_name : str
                Name of the model to be used for inference.
        '''
        self.model_name = model_name
        self.filepath = f'{os.getcwd()}/trained_models/{self.model_name}'
        if not (self.filepath[-3:] == '.pt' or self.filepath[-4:] == '.pth'):
            self.filepath = self.filepath + '.pt'
        
        self.model = torch.load(filepath)

    def infer(self, inp):
        '''
        Infers from the given input.

        Arguments
        ---------
            input : str
                Input for inference.
        '''
        return self.model.infer(inp)