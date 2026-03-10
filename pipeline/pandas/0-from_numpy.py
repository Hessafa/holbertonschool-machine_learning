#!/usr/bin/env python3
"""Module that creates a pandas DataFrame from a NumPy array"""

import pandas as pd


def from_numpy(array):
    """Create a pandas DataFrame from a NumPy ndarray

    Args:
        array (numpy.ndarray): array to convert

    Returns:
        pandas.DataFrame: DataFrame with alphabetically labeled columns
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
