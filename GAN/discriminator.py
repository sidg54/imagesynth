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
        '''
        self.config = config
        
        self.batch_size = self.config.batch_size
        self.main = nn.Sequential(
            nn.Conv2d(),
            nn.LeakyReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.Conv2d(),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Performs a single forward pass through the network.
        '''
        return self.main(x)