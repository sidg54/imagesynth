# standard library imports
from __future__ import absolute_import

# third party imports
import torch
import torch.nn as nn


class Generator(nn.Module):
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
        self.config = config
        self.num_channels = self.config.num_channels
        self.z_size = self.config.z_size
        self.num_features = self.config.num_G_features

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_size, self.num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_features * 8, self.num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_features * 4, self.num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_features * 2, self.num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.num_features, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
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
        return self.main(x)