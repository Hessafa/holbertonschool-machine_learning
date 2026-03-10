#!/usr/bin/env python3
"""Module that converts selected DataFrame values to a numpy array"""

import pandas as pd


def array(df):
    """Select the last 10 rows of High and Close columns as a numpy array

    Args:
        df (pd.DataFrame): DataFrame containing High and Close columns

    Returns:
        numpy.ndarray: array of the selected values
    """
    return df[["High", "Close"]].tail(10).to_numpy()
