#!/usr/bin/env python3
"""Module that computes descriptive statistics of a DataFrame."""


def analyze(df):
    """Compute descriptive statistics for all columns except Timestamp.

    Args:
        df: pandas DataFrame

    Returns:
        pandas DataFrame: descriptive statistics of numeric columns
    """
    # Exclude Timestamp column if it exists
    df_numeric = df.drop(columns=['Timestamp'], errors='ignore')

    # Compute descriptive statistics
    stats = df_numeric.describe()

    return stats
