#!/usr/bin/env python3
"""Module that sets Timestamp as the DataFrame index"""


def index(df):
    """Set the Timestamp column as the index of the DataFrame

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: DataFrame with Timestamp as index
    """
    return df.set_index("Timestamp")
