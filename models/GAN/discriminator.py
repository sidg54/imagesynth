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
            nn.Conv2d(
                in_channels=self.num_channels, out_channels=self.num_features,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.num_features, out_channels=self.num_features * 2,
                kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(self.num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.num_features * 2, out_channels=self.num_features * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.num_features * 4, out_channels=self.num_features * 8,
                kernel_size=1, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Linear(self.num_features * 8 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

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
        output = self.out(output)
        output = output.view(-1, self.num_features * 4 * 4)
        output = self.sigmoid(output)
        return output