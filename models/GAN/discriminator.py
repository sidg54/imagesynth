# standard library imports
from __future__ import absolute_import

# third party imports
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    Class to define a discriminator for a GAN architecture.
    '''

    def __init__(self, config):
        '''
        Initializes an instance of the Discriminator class.

        Parameters
        ----------
            config : obj
                Configuration object with information needed to load data and train the network.
        '''
        super(Discriminator, self).__init__()

        self.config = config
        self.num_channels = self.config.num_channels
        self.z_size = self.config.z_size
        self.num_features = self.config.num_D_features
        
        self.batch_size = self.config.batch_size

        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels, self.num_features, 1, 2, 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv2d(self.num_features, self.num_features * 2, 1, 2, 1, bias=True),
            nn.BatchNorm2d(self.num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_features * 2, self.num_features * 4, 1, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_features * 4, self.num_features * 8, 1, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.num_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Performs a single forward pass through the network.

        Parameters
        ----------
            x : array_like
                Tensor input to be fed through the network.
        
        Returns
        -------
            array_like
                Transformed tensor output.
        '''
        output = self.main(x)
        return output