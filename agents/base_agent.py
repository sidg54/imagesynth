'''
'''
# standard library imports
import logging


class BaseAgent:
    '''
    '''

    def __init__(self, config):
        '''
        '''
        self.config = config
        self.logger = logging.getLogger('Agent')

    def load_checkpoint(self, file_name):
        '''
        '''
        raise NotImplementedError

    def save_checkpoint(self, file_name='checkpoint.pth', is_best=False):
        '''
        '''
        raise NotImplementedError

    def run(self):
        '''
        '''
        raise NotImplementedError

    def train(self):
        '''
        '''
        raise NotImplementedError

    def train_one_epoch(self):
        '''
        '''
        raise NotImplementedError

    def validate(self):
        '''
        '''
        raise NotImplementedError

    def finalize(self):
        '''
        '''
        raise NotImplementedError
