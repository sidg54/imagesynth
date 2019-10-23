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
        super(Generator, self).__init__()

        self.config = config
        self.num_channels = self.config.num_channels
        self.z_size = self.config.z_size
        self.num_features = self.config.num_G_features

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.z_size, out_channels=self.num_features * 8, 
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.num_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=self.num_features * 8, out_channels=self.num_features * 4,
                kernel_size=1, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=self.num_features * 4, out_channels=self.num_features * 2,
                kernel_size=1, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=self.num_features * 2, out_channels=self.num_features,
                kernel_size=1, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=self.num_features, out_channels=self.num_channels,
                kernel_size=1, stride=1, padding=1, bias=False
            ),
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