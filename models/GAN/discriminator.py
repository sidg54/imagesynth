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
                Configuration object with information needed to load data and train the network.
        '''
        super(Discriminator, self).__init__()

        self.config = config
        self.num_channels = self.config.num_channels
        self.num_features = self.config.num_D_features
        self.kernel_size = self.config.kernel_size

        # layers
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.num_features,
            kernel_size=self.kernel_size,
            stride=1, padding=1, bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=self.num_features, out_channels=self.num_features * 2,
            kernel_size=self.kernel_size,
            stride=1, padding=1, bias=False
        )
        self.bnorm1 = nn.BatchNorm2d(self.num_features * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=self.num_features * 2, out_channels=self.num_features * 4,
            kernel_size=self.kernel_size,
            stride=1, padding=1, bias=False
        )
        self.bnorm2 = nn.BatchNorm2d(self.num_features * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(
            in_channels=self.num_features * 4, out_channels=self.num_features * 8,
            kernel_size=self.kernel_size,
            stride=1, padding=1, bias=False
        )
        self.bnorm3 = nn.BatchNorm2d(self.num_features * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(
            in_channels=self.num_features * 8, out_channels=1, kernel_size=self.kernel_size,
            stride=1, padding=0, bias=False
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
        print(output.size())

        output = self.conv2(output)
        output = self.bnorm1(output)
        output = self.lrelu2(output)
        print(output.size())

        output = self.conv3(output)
        output = self.bnorm2(output)
        output = self.lrelu3(output)
        print(output.size())

        output = self.conv4(output)
        output = self.bnorm3(output)
        output = self.lrelu4(output)
        print(output.size())
        
        output = self.conv5(output)
        output = self.sigmoid(output)
        print(output.size())
        output = output.view(-1)
        return output