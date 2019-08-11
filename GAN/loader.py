# standard library imports
from __future__ import absolute_import


class Loader:
    '''
    '''

    def __init__(self, config):
        '''
        '''
        self.config = config
        
        self.num_workers = self.config.num_workers
        self.dataset = self.config.dataset