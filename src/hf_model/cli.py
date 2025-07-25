import argparse
from typing import Iterable

parser = argparse.ArgumentParser()
parser.prog = "Model Server"

def parse_one_argument():
    r"""Decorator for converting the function into a one argument parse
    Takes in function name as argument name, help as function docstring
    Example Usage:
    ```python
    @parse_one_argument()
    def simple():
        '''This is a simple function'''
        print("Hello")
    ```
    """
    def inner_parser(func):
        parser.add_argument(func.__name__, help=func.__doc__)
        if func.__name__ in parser.parse_args():
            func()

    return inner_parser

def parse_one_argument_multiname(extraname):
    r"""Decorator for converting the function into a one argument parse
    Can take a extraname
    Takes in function name as argument name, help as function docstring
    Example Usage:
    ```python
    @parse_one_argument_multiname(extraname="-s")
    def simple():
        '''This is a simple function'''
        print("Hello")
    ```
    """
    def inner_parser(func):
        if not isinstance(extraname, str):
            raise argparse.ArgumentError(
                argument=None, 
                message="Do not pass extraname as anything other than a single string"
            )
        parser.add_argument(func.__name__, help=func.__doc__)
        func()
    return inner_parser
