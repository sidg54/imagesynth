r'''
Contains functions to deal with configuration 
objects used for training.
'''
# standard library imports
from __future__ import print_function, absolute_import
import os
from datetime import datetime

# third party imports
from yaml import safe_load, safe_dump
from termcolor import colored, cprint
from easydict import EasyDict as edict

# internal imports
from utils.utils import print_line


DATE_FORMAT = '%d-%m-%Y___%H-%M'


def process_config_file(config_file):
    '''
    Retrieves a yaml configuration file and processes
    it so that it can be used in other aspects of the
    project.

    Arguments
    ---------
        config_file : str
            Name of the configuration file to be parsed.
    
    Returns
    -------
        config : obj (dict)
            Configuration object (a dict)
    '''
    config = yaml_to_dict(yaml_file=config_file)
    try:
        print_line(color='green')
        cprint(f'Running: {config.experiment_name}', 'green')
        print_line(color='green')
        print('Configuration:\n')
        print_config(config)
        print_line(color='green')
        return config
    except AttributeError:
        print(
            colored('ERROR ::', 'red'),
            ' experiment_name not specified in YAML configuration file'
        )
        exit(-1)

def yaml_to_dict(yaml_file):
    '''
    Given a yaml file path, retrieves the file and
    parses to an object that can be used in python.

    Arguments
    ---------
        yaml_file : str
            Name of file to be parsed.
    
    Returns
    -------
        config : obj (dict)
            Configuration object (a dict).
    '''
    print(yaml_file)
    with open(yaml_file, 'r') as f:
        try:
            config_dict = safe_load(f)
            config = edict(config_dict)
            return config
        except ValueError:
            print('ERROR : YAML config file formatted improperly')
            exit(-1)


def log_config_file(config, seed, duration, start_time, end_time):
    '''
    Saves the results of the experiment and various other
    information regarding training to a file for future reference.

    Arguments
    ---------
        config : obj
            Configuration object used for training.
        seed : int
            Seed used in training.
        duration : str
            Duration of training.
        start_time : str
            Time of training start.
        end_time :  str
            Time of training end.
    '''
    save_dict = dict()
    save_dict.update(config)

    saved_dict = saved_dict.update({
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'used_seed': seed,
    })

    now = datetime.now()
    filepath = f'/experiment_logs/{config.experiment_name}_{now.strftime(DATE_FORMAT)}.yml'
    saved_filepath = filepath
    filepath = os.getcwd() + filepath
    with open(filepath, 'w') as f:
        safe_dump(saved_dict, f, default_flow_style=False)
    
    print(f'EXPERIMENT INFORMATION SAVED TO {saved_filepath}')
    print(f'FILE CONTENTS:\n\t{print(saved_dict)}')


def print_config(config):
    ''' Prints the configuration object in an easily readable format. '''
    for key, val in config.items():
        print(f'\t{key}  :  {val}')