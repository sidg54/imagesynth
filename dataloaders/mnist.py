r'''
Loads MNIST data.
'''
# standard library imports
from __future__ import absolute_import
from os import path, getcwd

# third party imports
import torch
from torchvision import datasets, transforms

# internal imports
from .dataloader import BaseDataLoader


class MNISTDataLoader(BaseDataLoader):
    '''
    Class to load MNIST data.
    '''

    def __init__(self, num_workers=4, batch_size=64, device='cpu'):
        '''
        Initializes an instance of the MNISTDataLoader class.

        Arguments
        ---------
            num_workers : (int, optional, default=4)
                Number of workers for dataloading.
            batch_size : (int, optional, default=64)
                Size of batch for loading.
            device : (str, optional, default="cpu"
                Device to use for computation.
        '''
        super().__init__(num_workers, batch_size, device)

        self.download = True
        if path.exists(f'{getcwd}/data/MNIST'):
            self.download = False
        
    def load_data(self):
        ''' Load the MNIST data. '''
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=f'./data/', train=True, download=self.download,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ), batch_size=64, shuffle=True, num_workers=4
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=f'./data/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ), batch_size=64, shuffle=True, num_workers=4
        )

        return train_loader, test_loader