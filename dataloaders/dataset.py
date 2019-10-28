r'''
Module to hold base dataloading class.
'''
# standard library imports
import os

# third party imports
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    '''
    Base dataloading class.
    '''

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
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.images = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join
    
    def load_data(self):
        ''' Loads data. '''
        raise NotImplementedError