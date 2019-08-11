r'''
'''
# standard library imports
from __future__ import absolute_import, print_function
import random

# third party imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# internal imports
from generator import Generator
from discriminator import Discriminator
from loader import Loader


class GAN:
    '''
    Class to define a GAN architecture for image synthesis.
    '''

    def __init__(self, config):
        '''
        Initializes an instance of the GAN class.
        '''
        self.config = config

        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.num_epochs = self.config.num_epochs
        self.input_size = self.config.input_size
        self.num_layers = self.config.num_layers
        
        self.G = Generator(config)
        self.D = Discriminator(config)

        self.G_optim = optim.SGD(lr=self.learning_rate)
        self.D_optim = optim.SGD(lr=self.learning_rate)

        self.loader = Loader(config)
    
    def train_one_epoch(self):
        '''
        Run a single training loop.
        '''
        pass

    def train(self):
        '''
        Run training.
        '''
        pass

    def finish_training(self):
        '''
        Finish training by graphing the network,
        saving G and D and show the results of training.
        '''
        pass