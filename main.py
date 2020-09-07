r'''
Main module for running any model in the project.
Must be run with Python3 since Python2 is losing
support by January 1, 2020 (a few months after
this is being written).

To run: 
    `python3 main.py <config_file_name>`

Author
------
Siddarth Ganguri
October 2019
'''
# standard library imports
import sys
from os import getcwd

# internal imports
from models import *
from utils.config import process_config_file
from utils.custom_parser import CustomParser
from models.gan import *
from models.base_agent import Agent


def main():
    '''
    Main call to run the given config
    file and its associated model, dataset, etc.
    '''
    # run with python 3
    if sys.version_info[0] < 3:
        print('ERROR :: Running with Python 2, please retry with Python 3')
        sys.exit(1)
    
    parser = CustomParser()
    parser.add_argument(
        'config',
        metavar='config_yml_file',
        default='None',
        help='Name of the configuration file to use. Written in YAML.'
    )

    # get config file specified in CL
    args = parser.parse_args()
    config_file = getcwd() + '/configs/' + args.config + '.yml'
    config = process_config_file(config_file)

    # ensure visual display is accounted for
    if 'visual' not in config:
        config['visual'] = False

    # sanity checks for loading agents from config name
    try:
        model_file = globals()[config.model_name]
    except Exception as e:
        print(f'Encountered an error loading the model with the given model name: {config.model_name}')
        raise e
    try:
        model = model_file(config)
    except Exception as e:
        print(f'Encountered an error initializing the model object')
        raise e

    # model must be of type Agent
    if type(model) != Agent:
        raise ValueError('model must be of type Agent (see models/base_agent.py)')

    # run with high-level error-handling
    try:
        model.run()
    except Exception as e:
        print('Error running training for the model')
        raise e
    try:
        model.finish_training()
    except Exception as e:
        print('Error finalizing training')
        raise e


if __name__ == '__main__':
    main()