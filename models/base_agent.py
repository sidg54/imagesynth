r'''
Module defining a base Agent class that all
agents in the project inherit from.

Author
------
Siddarth Ganguri
November 2019
'''
# standard library imports
import logging
import random

# third party imports
import torch
from torch.utils.tensorboard import SummaryWriter

# internal imports
from utils.utils import show_gpu


class Agent:
    ''' Class defining a base agent. '''

    def __init__(self, config):
        '''
        Initializes an instance of the Agent class.

        Arguments
        ---------
            config : dict
                Configuration object with information needed to load data and train.
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
            self.num_layers = int(self.config.num_layers)
        except ValueError:
            raise ValueError('num_layers in config cannot be parsed to int')

        try:
            self.num_epochs = int(self.config.num_epochs)
        except ValueError:
            raise ValueError('num_epochs in config cannot be parsed to int')

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

        self.cuda = torch.cuda.is_available() and self.config.cuda

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
        
        load_name = globals()[config.loader]
        self.loader = load_name(
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            device=self.device
        )
        self.train_loader, self.test_loader = self.loader.load_data()

        # set start and end time to None when initializing the class
        # this ensures calling the wrong method results in an error
        self.start_time = None
        self.end_time = None
    
    def checkpoint(self, state, filename='checkpoint.pth'):
        '''
        Saves the state dict of the network(s), the optimizers,
        and various other information to the given filename.

        Arguments
        ---------
            filename : str
                Name of the file to save the state to.
        '''
        # sanity check: state must be a dict
        assert type(state) == dict
        torch.save(state, self.config.checkpoint_dir + filename)
    
    def init_weights(self, net):
        '''
        Initializes custom weights for the given network.

        Arguments
        ---------
            net : obj
                Network to initialize weights for.
        '''
        raise NotImplementedError
    
    def set_train(self):
        ''' Sets the network(s) to training mode. '''
        raise NotImplementedError
    
    def set_test(self):
        ''' Sets the network(s) to testing mode. '''
        raise NotImplementedError
    
    def train_one_epoch(self, epoch):
        ''' Run a single training loop. '''
        raise NotImplementedError
    
    def test_one_epoch(self, epoch):
        ''' Run a single testing loop. '''
        raise NotImplementedError
    
    def run(self):
        ''' Run training and testing loops. '''
        raise NotImplementedError
    
    def infer(self, x):
        '''
        Runs a single forward pass on the data to produce some novel
        generative output.

        Arguments
        ---------
            x : array_like
                Input data.
        
        Returns
        -------
            array_like
                Tensor of output data.
        '''
        raise NotImplementedError
    
    def finish_training(self):
        '''
        Finish training by graphing the network,
        and saving the results of training.
        '''
        raise NotImplementedError