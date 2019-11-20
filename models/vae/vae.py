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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# internal imports
from ..base_agent import Agent
from dataloaders.mnist import MNISTDataLoader


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