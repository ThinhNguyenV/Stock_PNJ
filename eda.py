import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Load the data
df = pd.read_csv('upload/PNJ.csv')

# Display basic information
print("\n--- Basic Information ---")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Check data types
print("\n--- Data Types ---")
print(df.dtypes)

# Check for missing values
print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {df.isnull().sum().sum()}")

# Convert Date/Time to datetime
print("\n--- Converting Date/Time ---")
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
print(f"Date range: {df['Date/Time'].min()} to {df['Date/Time'].max()}")
print(f"Total trading days: {df['Date/Time'].dt.date.nunique()}")

# Basic statistics
print("\n--- Basic Statistics ---")
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
print(df[numeric_cols].describe())

# Calculate daily returns
print("\n--- Daily Returns ---")
df['Daily_Return'] = df.groupby(df['Date/Time'].dt.date)['Close'].pct_change()
print(df['Daily_Return'].describe())

# Resample to daily data
print("\n--- Resampling to Daily Data ---")
# Set Date/Time as index
df.set_index('Date/Time', inplace=True)

# Create daily dataframe
daily_df = df.groupby(pd.Grouper(freq='D')).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"Daily data shape: {daily_df.shape}")
print(daily_df.head())

# Create weekly dataframe
print("\n--- Resampling to Weekly Data ---")
weekly_df = df.groupby(pd.Grouper(freq='W')).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"Weekly data shape: {weekly_df.shape}")
print(weekly_df.head())

# Create monthly dataframe
print("\n--- Resampling to Monthly Data ---")
monthly_df = df.groupby(pd.Grouper(freq='M')).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"Monthly data shape: {monthly_df.shape}")
print(monthly_df.head())

# Save resampled dataframes
os.makedirs('stock_analysis/data', exist_ok=True)
daily_df.to_csv('stock_analysis/data/pnj_daily.csv')
weekly_df.to_csv('stock_analysis/data/pnj_weekly.csv')
monthly_df.to_csv('stock_analysis/data/pnj_monthly.csv')

# Calculate volatility (standard deviation of returns)
print("\n--- Volatility Analysis ---")
daily_df['Daily_Return'] = daily_df['Close'].pct_change()
daily_volatility = daily_df['Daily_Return'].std()
annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days in a year
print(f"Daily volatility: {daily_volatility:.4f}")
print(f"Annualized volatility: {annualized_volatility:.4f}")

# Calculate rolling statistics
print("\n--- Rolling Statistics ---")
daily_df['20d_MA'] = daily_df['Close'].rolling(window=20).mean()
daily_df['50d_MA'] = daily_df['Close'].rolling(window=50).mean()
daily_df['200d_MA'] = daily_df['Close'].rolling(window=200).mean()
daily_df['20d_Volatility'] = daily_df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

print(daily_df[['Close', '20d_MA', '50d_MA', '200d_MA', '20d_Volatility']].tail())

# Save the processed daily data
daily_df.to_csv('stock_analysis/data/pnj_daily_processed.csv')

print("\nExploratory data analysis completed. Results saved to stock_analysis/data/")
