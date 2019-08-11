# standard library imports
from __future__ import absolute_import

# third party imports
import torchvision


class Loader:
    '''
    Loads a dataset (from torchvision).
    '''

    def __init__(self, config):
        '''
        Initializes an instance of the Loader class.

        Parameters
        ----------
            config : obj
                Configuration object with information needed to load data and train the network.
        '''
        self.config = config
        
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.dataset_name = self.config.dataset_name

    def get_data_loader(self):
        '''
        '''
