r'''
Module to hold base dataloading class.
'''
# third party imports
import torch


class DataLoader:
    '''
    Base dataloading class.
    '''

    def __init__(self, num_workers=4, batch_size=64, device='cpu'):
        '''
        Initializes an instance of the MNISTDataLoader class.

        Arguments
        ---------
            num_workers : int   (default=4)
                Number of workers for dataloading.
            batch_size : int    (default=64)
                Size of batch for loading.
            device : str        (default="cpu")
                Device to use for computation.
        '''
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def load_data(self):
        ''' Loads data. '''
        raise NotImplementedError