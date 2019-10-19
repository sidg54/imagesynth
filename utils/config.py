r'''
Contains functions to deal with configuration 
objects used for training.
'''
# standard library imports
from __future__ import absolute_import

# third party imports
from yaml import safe_load, safe_dump


def log_config_file(config):
    '''
    Saves config file contents to logs/ directory as a YAML file.

    Arguments
    ---------
        config : dict
            Dictionary to be saved.
    '''
    pass