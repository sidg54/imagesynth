r'''
GAN agent to generate hand-written digits.

Author
------
Siddarth Ganguri
October 2019
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
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from IPython.display import HTML

# internal imports
from .generator import Generator
from .discriminator import Discriminator
from dataloaders.mnist import MNISTDataLoader
from utils.utils import show_gpu, imshow, plot_classes_preds, images_to_probs
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
        self.writer = SummaryWriter(f'experiment_logs/{self.config.experiment_name}')

        self.cur_epoch = 0
        self.cur_iteration = 0

        # below, we set a bunch of hyperparameters / class variables
        # and ensure they can be parsed to float or int (depending)
        try:
            self.print_every = int(self.config.print_every)
        except ValueError:
            raise ValueError('print_every in config cannot be parsed to int')

        try:
            self.batch_size = int(self.config.batch_size)
        except ValueError:
            raise ValueError('batch_size in config cannot be parsed to int')

        try:
            self.learning_rate = float(self.config.learning_rate)
        except ValueError:
            raise ValueError('learning_rate in config cannot be parsed to float')

        try:
            self.beta1 = float(self.config.beta1)
        except ValueError:
            raise ValueError('beta1 in config cannot be parsed to float')

        try:
            self.beta2 = float(self.config.beta2)
        except ValueError:
            raise ValueError('beta2 in config cannot be parsed to float')
        
        try:
            self.num_layers = int(self.config.num_layers)
        except ValueError:
            raise ValueError('num_layers in config cannot be parsed to int')

        try:
            self.num_G_features = int(self.config.num_G_features)
        except ValueError:
            raise ValueError('num_G_features in config cannot be parsed to int')

        try:
            self.num_D_features = int(self.config.num_D_features)
        except ValueError:
            raise ValueError('num_D_features in config cannot be parsed to int')

        try:
            self.z_size = int(self.config.z_size)
        except ValueError:
            raise ValueError('z_size in config cannot be parsed to int')

        try:
            self.num_epochs = int(self.config.num_epochs)
        except ValueError:
            raise ValueError('num_epochs in config cannot be parsed to int')

        # Use binary cross entropy loss for training.
        self.criterion = nn.BCELoss()

        self.real_label = 1
        self.fake_label = 0
        self.fixed_noise = Variable(torch.randn(self.batch_size, self.num_G_features))

        self.cuda = torch.cuda.is_available() and self.config.cuda

        try:
            self.ngpu = int(self.config.ngpu)
        except ValueError:
            raise ValueError('ngpu in config cannot be parsed to int')

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
            print('USING CUDA FOR TRAINING')
            show_gpu()
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.seed)
            self.logger.info('USING CPU FOR TRAINING')
            print('USING CPU FOR TRAINING')
        
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

        # save image grid to SummaryWriter
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('mnist_image_grid', img_grid)

        # set start and end time to None when initializing the class
        # this ensures calling the wrong method results in an error
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
    
    def set_train(self):
        ''' Sets the networks to training mode. '''
        self.G.train()
        self.D.train()
    
    def set_test(self):
        ''' Sets the networks to testing mode. '''
        self.G.eval()
        self.D.eval()
    
    def train_one_epoch(self, epoch):
        ''' Run a single training loop. '''
        # Set G and D to training so gradient updates occur
        self.set_train()

        # Training loop
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            # send data and target tensors to device
            data, target = data.to(self.device), target.to(self.device)
            ###############################################################
            # Update Discriminator to maximize log(D(x)) + log(1 - D(G(z)))
            ###############################################################
            # First, train with real
            # Zero gradients and begin training
            self.D_optim.zero_grad()
            label = torch.full( ( data.size(0), ), self.real_label, device=self.device)
            # Perform a single forward pass through D
            output = self.D(data)
            errD_real = self.criterion(output, label)
            # Calculate gradients
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(self.batch_size, self.z_size, 1, 1, device=self.device)
            fake = self.G(noise)
            print(dir(self.D))
            print(self.D.state_dict())
            # Fill with fake
            label.fill_(self.fake_label)
            output = self.D(fake.detach())
            # Backwards to update gradient weights for fake data
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute full error for D and take a single step w/ the optimizer for D
            errD = errD_real + errD_fake
            self.D_optim.step()


            #######################################
            # Update Generator to max log(D(G(z)))
            #######################################
            # Zero gradients for G
            self.G.zero_grad()
            label.fill_(real_label)
            # Retrieve D's output
            output = D(fake)
            # Calculate loss and update gradients
            errG = self.criterion(output, label)
            errG.backward()
            # Compute error for G and take a single step w/ the optimizer for G
            D_G_z2 = output.mean().item()
            self.G_optim.step()

            if epoch % self.print_every == 0:
                # save to SummaryWriter
                self.writer.add_scalar('G training loss',
                    errG.item() / 1000, epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('D training loss',
                    errD.item() / 1000, epoch * len(self.train_loader) + batch_idx)

                # print
                print('=================================================================')
                print(f'[{epoch}/{self.num_epochs}]\n[{batch_idx}/{len(self.train_loader)}]\n \
                    Loss D: {errD.item()}\nLoss G: {errG.item()}\n \
                        D(x): {D_x}\nD(G(z)): {D_G_z1} / {D_G_z2}')
                print('=================================================================')
            
            # Save losses for later plotting and analysis
            self.G_train_losses.append(errG.item())
            self.D_train_losses.append(errD.item())

            if self.cur_iteration % 500 == 0 or ((epoch == num_epochs - 1) and (batch_idx == len(self.train_loader)-1)):
                with torch.no_grad():
                    fake = self.G(self.fixed_noise).detach().cpu()
                self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            self.cur_iteration += 1

    def test_one_epoch(self, epoch):
        ''' Run a single testing loop. '''
        # Set G and D to eval so gradients updates do not occur
        self.set_test()
            
    def run(self):
        ''' Run training and testing loops. '''
        self.start_time = time.time()

        # if using multi-gpu, use DataParallel on G and D
        if self.device.type == 'cuda' and self.ngpu > 1:
            device_ids = list(range(self.ngpu))
            self.G = nn.DataParallel(self.G, device_ids=device_ids)
            self.D = nn.DataParallel(self.D, device_ids=device_ids)

        # apply weight initialization for G and D
        self.G.apply(self.init_weights)
        self.D.apply(self.init_weights)

        # track a list of images
        self.img_list = []

        # track train and test loss for graphing
        self.G_train_losses = []
        self.D_train_losses = []
        self.G_test_losses  = []
        self.D_test_losses  = []

        # run the training loop, alternating between train and test
        for epoch in range(self.num_epochs):
            # run a single training epoch
            self.train_one_epoch(epoch)

            # run a single test epoch
            self.test_one_epoch(epoch)

            # flip labels halfway through training.
            if epoch == math.floor(self.num_epochs / 2):
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
        if self.start_time is None:
            raise ValueError('start_time cannot be None')
        if self.end_time is None:
            raise ValueError('end_time cannot be None')

        self.set_test()
        gen_img = self.G(x)
        return gen_img

    def finish_training(self):
        '''
        Finish training by graphing the network,
        saving G and D and show the results of training.
        '''
        # ensure training occurred by checking start and end times
        # class variables are not set to None
        if self.start_time is None:
            raise ValueError('start_time cannot be None')
        if self.end_time is None:
            raise ValueError('end_time cannot be None')

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
        self.writer.close()

        # Plot training loss for G and D
        plt.figure(figsize=(10,5))
        plt.title('Generator and Discriminator Loss During Training')
        plt.plot(self.G_train_losses, label='G')
        plt.plot(self.D_train_losses, label='D')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Visualizing the Generator's progress through training.
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())

        # Plot real images
        real_batch = next(iter(self.train_loader))
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title('Real Images')
        plt.imshow(np.transpose(
            vutils.make_grid(
                real_batch[0].to(self.device)[:64],
                padding=5,
                normalize=True
            ).cpu(),(1,2,0))
        )

        # Plot Fake image generated during the last epoch
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title('Fake Images')
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        plt.show()