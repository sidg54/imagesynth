r'''
Basic discriminator for a GAN.

Author
------
Siddarth Ganguri
October 2019
'''
# standard library imports
from __future__ import absolute_import

# third party imports
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    ''' Class to define a discriminator network for a GAN architecture. '''

    def __init__(self, config):
        '''
        Initializes an instance of the Discriminator class.

        Parameters
        ----------
            config : obj
                Configuration object with information
                needed to train the network.
        '''
        super(Discriminator, self).__init__()

        # Configuration
        self.config = config
        self.num_channels = self.config.num_channels
        self.num_features = self.config.num_D_features
        self.kernel_size = self.config.kernel_size

        #### Layers
        # First conv block
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.num_features,
            kernel_size=self.kernel_size,
            stride=1, padding=0, bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # Second conv block
        self.conv2 = nn.Conv2d(
            in_channels=self.num_features, out_channels=self.num_features * 2,
            kernel_size=self.kernel_size,
            stride=1, padding=0, bias=False
        )
        self.bnorm1 = nn.BatchNorm2d(self.num_features * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        # Third conv block
        self.conv3 = nn.Conv2d(
            in_channels=self.num_features * 2, out_channels=self.num_features * 4,
            kernel_size=self.kernel_size,
            stride=2, padding=0, bias=False
        )
        self.bnorm2 = nn.BatchNorm2d(self.num_features * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        # Fourth conv block
        self.conv4 = nn.Conv2d(
            in_channels=self.num_features * 4, out_channels=self.num_features * 8,
            kernel_size=self.kernel_size,
            stride=2, padding=0, bias=False
        )
        self.bnorm3 = nn.BatchNorm2d(self.num_features * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # Output block
        self.conv5 = nn.Conv2d(
            in_channels=self.num_features * 8, out_channels=1, kernel_size=1,
            stride=2, padding=0, bias=False
        )
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
        output = self.conv1(x)
        output = self.lrelu1(output)

        output = self.conv2(output)
        output = self.bnorm1(output)
        output = self.lrelu2(output)

        output = self.conv3(output)
        output = self.bnorm2(output)
        output = self.lrelu3(output)

        output = self.conv4(output)
        output = self.bnorm3(output)
        output = self.lrelu4(output)
        
        output = self.conv5(output)
        output = self.sigmoid(output)

        output = output.view(-1)
        return output