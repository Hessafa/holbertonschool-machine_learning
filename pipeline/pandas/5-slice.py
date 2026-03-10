#!/usr/bin/env python3
"""Module that slices a DataFrame"""


def slice(df):
    """Extract specific columns and select every 60th row

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: sliced DataFrame
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
