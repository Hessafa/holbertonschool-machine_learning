#!/usr/bin/env python3
"""Module that loads data from a file into a pandas DataFrame"""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from a file as a pandas DataFrame

    Args:
        filename (str): file to load
        delimiter (str): column separator

    Returns:
        pandas.DataFrame: loaded DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
