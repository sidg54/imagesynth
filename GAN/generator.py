# standard library imports
from __future__ import absolute_import

# third party imports
import torch
import torch.nn as nn


class Generator(nn.Module):
    '''
    '''

    def __init__(self, config):
        '''
        '''
        self.config = config

        self.main = nn.Sequential(
            nn.ConvTranspose2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.Tanh()
        )
    
    def forward(self, x):
        '''
        '''
        return self.main(x)