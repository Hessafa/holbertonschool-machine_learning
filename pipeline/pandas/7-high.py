#!/usr/bin/env python3
"""Module that sorts a DataFrame by the High column"""


def high(df):
    """Sort the DataFrame by High price in descending order

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: sorted DataFrame
    """
    return df.sort_values(by="High", ascending=False)
