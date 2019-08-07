'''
'''
# standard library imports
from __future__ import absolute_import
import logging


class BaseDL:
    '''
    '''

    def __init__(self, config):
        '''
        '''
        self.config = config
        self.logger = logging.getLogger('DataLoader')

    def get_data(self):
        '''
        '''
        raise NotImplementedError
