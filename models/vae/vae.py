r'''
VAE agent to generate images.

Author
------
Siddarth Ganguri
November 2019
'''
# standard library imports
from __future__ import absolute_import, print_function
import random
import logging
import time
import math
from copy import copy

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# internal imports
from ..base_agent import Agent
from dataloaders.mnist import MNISTDataLoader
from utils.utils import show_gpu, imshow, plot_classes_preds, images_to_probs
from utils.config import log_config_file, print_config


class VAE(Agent):
    ''' Class to define a VAE architecture for image synthesis. '''

    def __init__(self, config):
        '''
        Initializes an instance of the VAE class.

        Arguments
        ---------
            config : dict
                Configuration dict with information needed for data loading and training.
        '''
        super(Agent, self).__init__(config)

        self.criterion = nn.BCELoss()       # TODO: might not be best

        enc_name = globals()[self.config.encoder]
        dec_name = globals()[self.config.decoder]
        self.encoder = enc_name(self.config).to(self.device)
        self.decoder = dec_name(self.config).to(self.device)

        self.enc_optim = optim.Adam(
            self.encoder.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        self.dec_optim = optim.Adam(
            self.decoder.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
    
    def init_weights(self, net):
        classname = net.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(net.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(net.weight.data, 1.0, 0.02)
            nn.init.constant_(net.bias.data, 0)
    
    def set_train(self):
        self.encoder.train()
        self.decoder.train()
    
    def set_test(self):
        self.encoder.eval()
        self.decoder.eval()
    
    def train_one_epoch(self, epoch):
        self.set_train()
        # TODO:
    
    def test_one_epoch(self, epoch):
        self.set_test()
        # TODO:
    
    def run(self):
        self.start_time = time.time()

        # if using multi-gpu, use DataParallel on encoder and decoder
        if self.device_type == 'cuda' and self.ngpu > 1:
            device_ids = list(range(self.ngpu))
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids)
            self.decoder = nn.DataParallel(self.decoder, device_ids=device_ids)
        
        # apply weight initialization for encoder and decoder
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        # keeps track of a list of images
        self.img_list = []

        # track train and test loss for graphing
        self.hist = {}
        self.hist['encoder_train_losses'] = []
        self.hist['decoder_train_losses'] = []
        self.hist['encoder_test_losses'] = []
        self.hist['decoder_test_losses'] = []

        # run the training loop, alternating between train and test
        for epoch in range(self.num_epochs):
            # run a single training epoch
            self.train_one_epoch(epoch)

            # run a single test epoch
            self.test_one_epoch(epoch)

        # calculate duration of training and set as class attribute
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def infer(self, x):
        if self.start_time is None:
            raise ValueError('start_time cannot be None')
        if self.end_time is None:
            raise ValueError('end_time cannot be None')

        self.set_test()
        generated_image = self.decoder(self.encoder(x))
        return generated_image
    
    def finish_training(self):
        # ensure training occurred by checking start and end times
        # class variables are not set to None
        if self.start_time is None:
            raise ValueError('start_time cannot be None')
        if self.end_time is None:
            raise ValueError('end_time cannot be None')
        if self.duration is None:
            raise ValueError('duration cannot be None at the end of training.')

        # all info relevant from training placed in dict
        new_config_info = {
            'training_duration': self.duration,
            'final_loss': self.criterion,
            'cuda_used': self.cuda,
            'seed_used': self.seed,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'hist': self.hist,
        }

        # update and log dict
        self.config.update(new_config_info)
        print_config(config=self.config)
        log_config_file(config=self.config)

        # plot training loss for encoder and decoder
        plt.figure(figsize=(10,5))
        plt.title('Encoder and Decoder Loss During Training')
        plt.plot(self.hist['encoder_train_losses'], label='encoder')
        plt.plot(self.hist['decoder_train_losses'], label='decoder')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # visualizing the generated data samples through training
        # TODO: