#!/usr/bin/env python3
"""Module that fills missing values in the DataFrame"""


def fill(df):
    """Fill missing values in a DataFrame with specific rules

    Args:
        df: pandas DataFrame

    Returns:
        pandas.DataFrame: modified DataFrame
    """
    # 1️⃣ Remove the Weighted_Price column
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # 2️⃣ Fill missing Close with previous row
    df["Close"] = df["Close"].fillna(method="ffill")

    # 3️⃣ Fill missing High, Low, Open with same row's Close
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # 4️⃣ Fill missing volumes with 0
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df
