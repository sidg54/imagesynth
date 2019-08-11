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
from torch.autograd import Variable
from torch.backends import cudnn

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

        Parameters
        ----------
            config : obj
                Configuration object with information needed to load data and train the network.
        '''
        self.config = config

        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.num_epochs = self.config.num_epochs
        self.input_size = self.config.input_size
        self.num_layers = self.config.num_layers
        self.num_G_features = self.config.num_G_features
        self.num_D_features = self.config.num_D_features
        
        self.G = Generator(config)
        self.D = Discriminator(config)

        self.G_optim = optim.SGD(
            self.G.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        self.D_optim = optim.SGD(
            self.D.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )

        self.loss = nn.BCELoss()

        self.real_label = 1
        self.fake_label = 0

        self.fixed_noise = Variable(torch.randn(self.batch_size, self.num_G_features))

        self.loader = Loader(config)
        self.train_loader, self.test_loader = self.loader.get_loaders()
    
    def train_one_epoch(self):
        '''
        Run a single training loop.
        '''
        self.G.train()
        self.D.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.D_optim.zero_grad()
            output = self.D(data)

    def test_one_epoch(self):
        '''
        Run single testing loop.
        '''
        self.G.eval()
        self.D.eval()
            
    def run(self):
        '''
        Run training and testing loops.
        '''
        train_loss = []
        test_loss = []
        for epoch in range(self.num_epochs):
            self.train()
            self.test()

    def finish_training(self):
        '''
        Finish training by graphing the network,
        saving G and D and show the results of training.
        '''
        pass