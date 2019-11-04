# standard library imports
import sys
from argparse import ArgumentParser


class CustomParser(ArgumentParser):
    '''
    Wrapper class for argparse.ArgumentParser
    to print out details when an error occurs
    attempting to parse CL input.
    '''

    def error(self, message):
        '''
        When an error occurs parsing from CL,
        write message to stdout and print the help statements.
        '''
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        # generally use sys.exit(2) for CL syntax errors
        # according to https://docs.python.org/3/library/sys.html#sys.exit
        sys.exit(2)