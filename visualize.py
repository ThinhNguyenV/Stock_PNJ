import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter
import mplfinance as mpf

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Create output directory for visualizations
os.makedirs('stock_analysis/visualizations', exist_ok=True)

# Load the processed daily data
print("Loading processed daily data...")
daily_df = pd.read_csv('stock_analysis/data/pnj_daily_processed.csv')
daily_df['Date/Time'] = pd.to_datetime(daily_df['Date/Time'])
daily_df.set_index('Date/Time', inplace=True)

# Load weekly and monthly data
weekly_df = pd.read_csv('stock_analysis/data/pnj_weekly.csv')
weekly_df['Date/Time'] = pd.to_datetime(weekly_df['Date/Time'])
weekly_df.set_index('Date/Time', inplace=True)

monthly_df = pd.read_csv('stock_analysis/data/pnj_monthly.csv')
monthly_df['Date/Time'] = pd.to_datetime(monthly_df['Date/Time'])
monthly_df.set_index('Date/Time', inplace=True)

# Function to format y-axis with commas
def format_with_commas(x, pos):
    return '{:,.0f}'.format(x)

# 1. Daily Price Chart with Moving Averages
print("Creating daily price chart with moving averages...")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['Close'], label='Close Price', color='blue', alpha=0.7)
plt.plot(daily_df.index, daily_df['20d_MA'], label='20-Day MA', color='red', alpha=0.7)
plt.plot(daily_df.index, daily_df['50d_MA'], label='50-Day MA', color='green', alpha=0.7)
plt.plot(daily_df.index, daily_df['200d_MA'], label='200-Day MA', color='purple', alpha=0.7)
plt.title('PNJ Daily Close Price with Moving Averages (2018-2020)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/daily_price_with_ma.png', dpi=300)
plt.close()

# 2. Candlestick Chart (using mplfinance)
print("Creating candlestick chart...")
# Create a subset of the data for better visualization (last 6 months)
last_6m = daily_df.iloc[-180:].copy()
# Prepare data for mplfinance
ohlc_data = last_6m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Create candlestick chart
mpf.plot(ohlc_data, type='candle', style='yahoo', 
         title='PNJ Candlestick Chart (Last 6 Months)',
         ylabel='Price',
         volume=True,
         figsize=(14, 10),
         savefig='stock_analysis/visualizations/candlestick_chart.png')

# 3. Volume Chart
print("Creating volume chart...")
plt.figure(figsize=(14, 7))
plt.bar(daily_df.index, daily_df['Volume'], color='blue', alpha=0.7)
plt.title('PNJ Daily Trading Volume (2018-2020)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_commas))
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/volume_chart.png', dpi=300)
plt.close()

# 4. Price and Volume Combined
print("Creating price and volume combined chart...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Price plot
ax1.plot(daily_df.index, daily_df['Close'], label='Close Price', color='blue', alpha=0.7)
ax1.set_title('PNJ Price and Volume (2018-2020)', fontsize=16)
ax1.set_ylabel('Price', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Volume plot
ax2.bar(daily_df.index, daily_df['Volume'], color='blue', alpha=0.7)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Volume', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(FuncFormatter(format_with_commas))

plt.tight_layout()
plt.savefig('stock_analysis/visualizations/price_volume_combined.png', dpi=300)
plt.close()

# 5. Daily Returns Distribution
print("Creating daily returns distribution...")
plt.figure(figsize=(14, 7))
sns.histplot(daily_df['Daily_Return'].dropna(), kde=True, bins=100)
plt.title('PNJ Daily Returns Distribution (2018-2020)', fontsize=16)
plt.xlabel('Daily Return', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/daily_returns_distribution.png', dpi=300)
plt.close()

# 6. Rolling Volatility
print("Creating rolling volatility chart...")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['20d_Volatility'], label='20-Day Rolling Volatility (Annualized)', color='red')
plt.title('PNJ 20-Day Rolling Volatility (2018-2020)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (Annualized)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/rolling_volatility.png', dpi=300)
plt.close()

# 7. Monthly Performance Heatmap
print("Creating monthly performance heatmap...")
# Create monthly returns
monthly_returns = daily_df['Close'].resample('M').last().pct_change()
monthly_returns = monthly_returns.to_frame()
monthly_returns['Year'] = monthly_returns.index.year
monthly_returns['Month'] = monthly_returns.index.month
monthly_pivot = monthly_returns.pivot_table(values='Close', index='Year', columns='Month')

plt.figure(figsize=(14, 8))
sns.heatmap(monthly_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
plt.title('PNJ Monthly Returns Heatmap (2018-2020)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/monthly_returns_heatmap.png', dpi=300)
plt.close()

# 8. Yearly Performance Comparison
print("Creating yearly performance comparison...")
yearly_returns = daily_df['Close'].resample('Y').last().pct_change()
plt.figure(figsize=(10, 6))
yearly_returns.plot(kind='bar', color='blue')
plt.title('PNJ Yearly Returns (2018-2020)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('stock_analysis/visualizations/yearly_returns.png', dpi=300)
plt.close()

print("All visualizations completed and saved to stock_analysis/visualizations/")
