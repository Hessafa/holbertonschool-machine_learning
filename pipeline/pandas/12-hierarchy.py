#!/usr/bin/env python3
"""Module that concatenates two DataFrames with a MultiIndex, 
with Timestamp first."""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenate bitstamp and coinbase DataFrames with Timestamp 
    as first index level for timestamps 1417411980 to 1417417980, 
    inclusive.

    Args:
        df1: pandas DataFrame (coinbase)
        df2: pandas DataFrame (bitstamp)

    Returns:
        pandas.DataFrame: concatenated DataFrame with MultiIndex 
        [Timestamp, source]
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select rows within the timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate df2 on top of df1 with keys
    df_concat = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Swap levels to make Timestamp the first level
    df_concat = df_concat.swaplevel(0, 1)

    # Sort by Timestamp to ensure chronological order
    df_concat = df_concat.sort_index(level=0)

    return df_concat
