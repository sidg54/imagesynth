# standard library imports
from __future__ import absolute_import
from os import listdir, getcwd

# third party imports
from torch.utils.data import DataLoader
from torchvision.dataset import MNIST
from torchvision.transforms import transforms


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
        self.shuffle = self.config.shuffle

    def get_dataset(self, train=False):
        '''
        Retrieves the desired dataset.

        Parameters
        ----------
            train : bool            (default=False)
                True if the dataset should be training set.
        
        Returns
        -------
            array_like
                Dataset for training or testing.
        '''
        dataset_path = getcwd() + '/data/' + dataset_name + '/'
        download = False
        if len(listdir(path=dataset_path)) == 0:
            download = True

        dataset = MNIST(
            root=mnist_path,
            train=train,
            download=download,
            transform=transforms([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (.3081,)
                )
            ])
        )
        return dataset

    def get_dataloaders(self):
        '''
        Retrieves a dataloader containing the images for training.

        Returns
        -------
            tuple : (array_like, array_like)
                Training dataloader, testing dataloader.
        '''
        train_dataset = self.get_dataset(train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        test_dataset = self.get_dataset(train=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        return train_dataloader, test_dataloader