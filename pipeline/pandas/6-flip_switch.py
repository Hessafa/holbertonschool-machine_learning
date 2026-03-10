#!/usr/bin/env python3
"""Module that flips and transposes a DataFrame"""


def flip_switch(df):
    """Sort the DataFrame in reverse chronological order and transpose it

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: transformed DataFrame
    """
    df = df.sort_index(ascending=False)
    return df.transpose()
