#!/usr/bin/env python3
"""Module that removes rows with NaN values in Close column"""


def prune(df):
    """Remove entries where Close has NaN values

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: modified DataFrame
    """
    return df.dropna(subset=["Close"])
