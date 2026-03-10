#!/usr/bin/env python3
"""Module that renames and formats DataFrame columns"""

import pandas as pd


def rename(df):
    """Rename Timestamp column and convert to datetime

    Args:
        df (pd.DataFrame): input DataFrame containing Timestamp column

    Returns:
        pd.DataFrame: modified DataFrame with Datetime and Close columns
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
