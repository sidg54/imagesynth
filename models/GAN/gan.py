r'''
GAN agent to generate hand-written digits.
'''
# standard library imports
from __future__ import absolute_import, print_function
import random
import logging
import time
import math
from copy import copy

# third party imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.backends import cudnn
from tqdm import tqdm

# internal imports
from .generator import Generator
from .discriminator import Discriminator
from dataloaders.mnist import MNISTDataLoader
from utils.utils import show_gpu
from utils.config import log_config_file, print_config


class GAN:
    ''' Class to define a GAN architecture for image synthesis. '''

    def __init__(self, config):
        '''
        Initializes an instance of the GAN class.

        Parameters
        ----------
            config : obj
                Configuration object with information needed to load data and train the network.
        '''
        self.config = config
        self.logger = logging.getLogger()

        self.cur_epoch = 0
        self.cur_iteration = 0

        self.batch_size = int(self.config.batch_size)
        self.learning_rate = float(self.config.learning_rate)
        self.beta1 = float(self.config.beta1)
        self.beta2 = float(self.config.beta2)
        self.num_epochs = self.config.num_epochs
        self.num_layers = int(self.config.num_layers)
        self.num_G_features = int(self.config.num_G_features)
        self.num_D_features = int(self.config.num_D_features)
        self.z_size = int(self.config.z_size)

        # Use binary cross entropy loss for training.
        self.criterion = nn.BCELoss()

        self.real_label = 1
        self.fake_label = 0
        self.fixed_noise = Variable(torch.randn(self.batch_size, self.num_G_features))

        self.cuda = torch.cuda.is_available() and self.config.cuda
        self.ngpu = int(self.config.ngpu)

        if self.config.seed < 0 or not self.config.seed:
            self.seed = random.randint(1, 10000)
        else:
            self.seed = self.config.seed
        self.logger.info('MANUAL SEED: ', self.seed)
        random.seed(self.seed)

        if self.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.device)
            torch.cuda.manual_seed_all(self.seed)
            self.logger.info('USING CUDA FOR TRAINING')
            show_gpu()
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.seed)
            self.logger.info('USING CPU FOR TRAINING')
        
        G_name = globals()[config.G]
        D_name = globals()[config.D]
        self.G = G_name(config).to(self.device)
        self.D = D_name(config).to(self.device)

        self.G_optim = optim.Adam(
            self.G.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        self.D_optim = optim.Adam(
            self.D.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )

        self.loader = MNISTDataLoader(
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        self.train_loader, self.test_loader = self.loader.load_data()

        self.start_time = None
        self.end_time = None
    
    def checkpoint(self, filename='checkpoint.pth'):
        '''
        Saves the state dict of D, G, the optimizers, and
        various other information to the given filename.

        Parameters
        ----------
            filename : str
                Name of the file to save the state to.
        '''
        state = {
            'epoch': self.cur_epoch,
            'iteration': self.cur_iteration,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'D_optim': self.D_optim.state_dict(),
            'fixed_noise': self.fixed_noise,
            'manual_seed': self.manual_seed
        }
        torch.save(state, self.config.checkpoint_dir + filename)
    
    def init_weights(self, net):
        '''
        Initializes custom weights for G and D.

        Arguments
        ---------
            net : obj
                Network to initialize weights for.
        '''
        classname = net.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(net.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(net.weight.data, 1.0, 0.02)
            nn.init.constant_(net.bias.data, 0)
    
    def flip_labels(self):
        ''' Flips the real and fake labels. '''
        new_fake = copy(self.real_label)
        new_real = copy(self.fake_label)
        self.real_label = new_real
        self.fake_label = new_fake
    
    def train_one_epoch(self, epoch):
        ''' Run a single training loop. '''
        # set G and D to training so gradient updates occur
        self.G.train()
        self.D.train()

        # training loop
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # send data and target tensors to device
            data, target = data.to(self.device), target.to(self.device)
            ###############################################################
            # Update Discriminator to maximize log(D(x)) + log(1 - D(G(z)))
            ###############################################################
            # first, train with real
            # zero gradients and begin training
            self.D_optim.zero_grad()
            label = torch.full((data.size(0),), self.real_label, device=self.device)
            # perform a single forward pass through D
            output = self.D(data)
            errD_real = self.criterion(output, label)
            # calculate gradients
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(self.batch_size, self.z_size, 1, 1, device=self.device)
            fake = self.G(noise)
            # fill with fake
            label.fill_(self.fake_label)
            output = self.D(fake.detach())
            # backwards to update gradient weights for fake data
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # compute full error for D and take a single step w/ the optimizer for D
            errD = errD_real + errD_fake
            self.D_optim.step()


            #######################################
            # Update Generator to max log(D(G(z)))
            #######################################
            # zero gradients for G
            self.G.zero_grad()
            label.fill_(real_label)
            # retrieve D's output
            output = D(fake)
            # calculate loss and update gradients
            errG = self.criterion(output, label)
            errG.backward()
            # compute error for G and take a single step w/ the optimizer for G
            D_G_z2 = output.mean().item()
            self.G_optim.step()

            if epoch % 50 == 0:
                print('=================================================================')
                print(f'[{epoch}/{self.num_epochs}]\n[{batch_idx}/{len(self.train_loader)}]\n \
                    Loss D: {errD.item()}\nLoss G: {errG.item()}\n \
                        D(x): {D_x}\nD(G(z)): {D_G_z1} / {D_G_z2}')
                print('=================================================================')

    def test_one_epoch(self, epoch):
        ''' Run single testing loop. '''
        # set G and D to eval so gradients updates do not occur
        self.G.eval()
        self.D.eval()
            
    def run(self):
        ''' Run training and testing loops. '''
        self.start_time = time.time()

        # if using multi-gpu, use DataParallel on G and D
        if self.device.type == 'cuda' and self.ngpu > 1:
            self.G = nn.DataParallel(self.G, list(range(self.ngpu)))
            self.D = nn.DataParallel(self.D, list(range(self.ngpu)))

        # apply weight initialization for G and D
        self.G.apply(self.init_weights)
        self.D.apply(self.init_weights)

        # track train and test loss for graphing
        train_loss = []
        test_loss = []
        # tqdm helps visualize loop progress
        for epoch in tqdm(range(self.num_epochs)):
            # run a single training epoch
            self.train_one_epoch(epoch)
            
            # run a single test epoch
            self.test_one_epoch(epoch)

            # flip labels halfway through training.
            if epoch % self.num_epochs == math.floor(self.num_epochs / 2):
                self.flip_labels()

        # calculate duration of training and set as class attribute
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def infer(self, x):
        '''
        Runs a single forward pass on the data to produce
        some novel generative output.

        Arguments
        ---------
            x : array_like
                Input data.
        
        Returns
        -------
            array_like
                Tensor of output data.
        '''
        self.G = self.G.eval()
        inference = self.G(x)
        return inference

    def finish_training(self):
        '''
        Finish training by graphing the network,
        saving G and D and show the results of training.
        '''
        # all info relevant from training placed in dict
        new_config_info = {
            'training_duration': self.duration,
            'final_loss': self.criterion,
            'cuda_used': self.cuda,
            'seed_used': self.seed,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'real_label': self.real_label,
            'fake_label': self.fake_label,
        }
        
        # update and log dict
        self.config.update(new_config_info)
        print_config(config=self.config)
        log_config_file(config=self.config)