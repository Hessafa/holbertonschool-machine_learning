#!/usr/bin/env python3
"""Visualize cryptocurrency data at daily intervals."""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop Weighted_Price column if it exists
df = df.drop(columns=['Weighted_Price'], errors='ignore')

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamps to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set Date as index
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 onward
df = df[df.index.year >= 2017]

# Resample daily and aggregate
daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the daily data
daily.plot(figsize=(14, 7))
plt.title('Daily Cryptocurrency Data (2017+)')
plt.ylabel('Price / Volume')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.show()

# Return the transformed DataFrame
daily
