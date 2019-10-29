# standard library imports
from __future__ import absolute_import

# third party imports
import torch
import torch.nn as nn


class Generator(nn.Module):
    '''
    Class to define a generator network for a GAN architecture.
    '''

    def __init__(self, config):
        '''
        Initializes an instance of the Generator class.

        Parameters
        ----------
            config : obj
                Configuration object with information needed to load data and train the network.
        '''
        super(Generator, self).__init__()

        self.config = config
        self.num_channels = self.config.num_channels
        self.num_features = self.config.num_G_features
        self.z_size = self.config.z_size
        self.kernel_size = self.config.kernel_size

        # layers
        self.convt1 = nn.ConvTranspose2d(
            in_channels=self.z_size, out_channels=self.num_features * 8, 
            kernel_size=self.kernel_size, stride=1, padding=0, bias=False
        )
        self.bnorm1 = nn.BatchNorm2d(self.num_features * 8)
        self.relu1 = nn.ReLU(True)

        self.convt2 = nn.ConvTranspose2d(
            in_channels=self.num_features * 8, out_channels=self.num_features * 4,
            kernel_size=self.kernel_size, stride=1, padding=1, bias=False
        )
        self.bnorm2 = nn.BatchNorm2d(self.num_features * 4)
        self.relu2 = nn.ReLU(True)

        self.convt3 = nn.ConvTranspose2d(
            in_channels=self.num_features * 4, out_channels=self.num_features * 2,
            kernel_size=self.kernel_size, stride=1, padding=1, bias=False
        )
        self.bnorm3 = nn.BatchNorm2d(self.num_features * 2)
        self.relu3 = nn.ReLU(True)

        self.convt4 = nn.ConvTranspose2d(
            in_channels=self.num_features * 2, out_channels=self.num_features,
            kernel_size=self.kernel_size, stride=1, padding=1, bias=False
        )
        self.bnorm4 = nn.BatchNorm2d(self.num_features),
        self.relu4 = nn.ReLU(True)

        self.convt5 = nn.ConvTranspose2d(
            in_channels=self.num_features, out_channels=self.num_channels,
            kernel_size=self.kernel_size, stride=1, padding=1, bias=False
        )
        self.tanh = nn.Tanh()
    
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
        output = self.convt1(x)
        output = self.bnorm1(output)
        output = self.relu1(output)

        output = self.convt2(output)
        output = self.bnorm2(output)
        output = self.relu2(output)
        
        output = self.convt3(output)
        output = self.bnorm3(output)
        output = self.relu3(output)

        output = self.convt4(output)
        output = self.bnorm4(output)
        output = self.relu4(output)

        output = self.convt5(output)
        output = self.tanh(output)

        return output