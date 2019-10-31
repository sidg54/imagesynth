r'''
Module to hold base dataloading class.
'''
# third party imports
import torch


class BaseDataLoader:
    ''' Base dataloading class. '''

    def __init__(self, csv_file, root_dir, num_workers=4, batch_size=64, device='cpu', transform=None):
        '''
        Initializes an instance of the MNISTDataLoader class.

        Arguments
        ---------
            csv_file : str
                Path to the csv file.
            root_dir : str
                Directory with all images.
            num_workers : (int, default=4)
                Number of workers for dataloading.
            batch_size : (int, default=64)
                Size of batch for loading.
            device : (str, default="cpu")
                Device to use for computation.
            transform : (callable, optional)
                Optional transforms to be applied on a sample.
        '''
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

    def load_data(self):
        ''' Loads data. '''
        raise NotImplementedError