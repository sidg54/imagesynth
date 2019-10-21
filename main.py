r'''
Main module for running any model in the project.
To run: 
    `main.py --config <config_file_name>`
'''
# standard library imports
from os import getcwd
from sys import version_info

# third party imports
from termcolor import colored, cprint

# internal imports
from models import *
from utils.config import process_config_file
from utils.custom_parser import CustomParser
from models.gan import *


def main():
    '''
    Main call to run the given config
    file and its associated model, dataset, etc.
    '''
    # run with python 3
    if version_info[0] < 3:
        colored('ERROR :: ', 'red')
        + 'Running with Python 2, please retry with Python 3'
    
    parser = CustomParser()
    parser.add_argument(
        'config',
        metavar='config_yml_file',
        default='None',
        help='Name of the configuration file to use. Written in YAML.'
    )

    args = parser.parse_args()
    config_file = getcwd() + '/configs/' + args.config + '.yml'
    config = process_config_file(config_file)

    # try:
    model_file = globals()[config.model_name]
    model = model_file(config)
    model.run()
    model.finalize()
    # except Exception as e:
    #     raise Exception('Could not run selected model')


if __name__ == '__main__':
    main()