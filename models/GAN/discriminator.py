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

        self.conv1 = nn.Conv2d(self.num_channels, self.num_features, 4, 2, 1, bias=False),
        self.l_relu1 = nn.LeakyReLU(0.3, inplace=True),

        self.conv2 = nn.Conv2d(self.num_features, self.num_features * 2, 4, 2, 1, bias=True),
        self.b_norm1 = nn.BatchNorm2d(self.num_features * 2),
        self.l_relu2 = nn.LeakyReLU(0.2, inplace=True),

        self.conv3 = nn.Conv2d(self.num_features * 2, self.num_features * 4, 4, 2, 1, bias=False),
        self.b_norm2 = nn.BatchNorm2d(self.num_features * 4),
        self.l_relu3 = nn.LeakyReLU(0.2, inplace=True),

        self.conv4 = nn.Conv2d(self.num_features * 4, self.num_features * 8, 4, 2, 1, bias=False),
        self.b_norm3 = nn.BatchNorm2d(self.num_features * 8),
        self.l_relu4 = nn.LeakyReLU(0.2, inplace=True),

        self.conv5 = nn.Conv2d(self.num_features * 8, 1, 4, 1, 0, bias=False),
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
        output = self.l_relu1(output)

        output = self.conv2(output)
        output = self.b_norm1(output)
        output = self.l_relu2(output)

        output = self.conv3(output)
        output = self.b_norm2(output)
        output = self.l_relu3(output)

        output = self.conv4(output)
        output = self.b_norm3(output)
        output = self.l_relu4(output)

        output = self.conv5(output)
        output = self.sigmoid(output)
        return output