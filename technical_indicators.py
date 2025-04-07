import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Create output directory for technical indicators
os.makedirs('stock_analysis/technical_indicators', exist_ok=True)

# Load the processed daily data
print("Loading processed daily data")
daily_df = pd.read_csv('stock_analysis/data/pnj_daily_processed.csv')
daily_df['Date/Time'] = pd.to_datetime(daily_df['Date/Time'])
daily_df.set_index('Date/Time', inplace=True)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    print(f"Calculating RSI with {window}-day window")
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    print(f"Calculating MACD with fast={fast}, slow={slow}, signal={signal}")
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    print(f"Calculating Bollinger Bands with {window}-day window and {num_std} standard deviations")
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, k_window=14, d_window=3):
    print(f"Calculating Stochastic Oscillator with K={k_window}, D={d_window}")
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    
    k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent, d_percent

# Function to calculate Average Directional Index (ADX)
def calculate_adx(data, window=14):
    print(f"Calculating ADX with {window}-day window")
    # Calculate True Range
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate Directional Movement
    data['DM+'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                          np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    data['DM-'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                          np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
    
    # Calculate Smoothed Averages
    data['ATR'] = data['TR'].rolling(window=window).mean()
    data['DI+'] = 100 * (data['DM+'].rolling(window=window).mean() / data['ATR'])
    data['DI-'] = 100 * (data['DM-'].rolling(window=window).mean() / data['ATR'])
    
    # Calculate Directional Index
    data['DX'] = 100 * (abs(data['DI+'] - data['DI-']) / (data['DI+'] + data['DI-']))
    
    # Calculate ADX
    data['ADX'] = data['DX'].rolling(window=window).mean()
    
    return data['ADX'], data['DI+'], data['DI-']

# Calculate RSI
daily_df['RSI'] = calculate_rsi(daily_df['Close'])

# Calculate MACD
daily_df['MACD'], daily_df['MACD_Signal'], daily_df['MACD_Histogram'] = calculate_macd(daily_df['Close'])

# Calculate Bollinger Bands
daily_df['BB_Middle'], daily_df['BB_Upper'], daily_df['BB_Lower'] = calculate_bollinger_bands(daily_df['Close'])

# Calculate Stochastic Oscillator
daily_df['Stoch_K'], daily_df['Stoch_D'] = calculate_stochastic(daily_df)

# Calculate ADX
daily_df['ADX'], daily_df['DI+'], daily_df['DI-'] = calculate_adx(daily_df)

# Save the data with technical indicators
daily_df.to_csv('stock_analysis/data/pnj_daily_with_indicators.csv')

# Plot RSI
print("Creating RSI plot")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['RSI'], color='blue')
plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
plt.fill_between(daily_df.index, y1=70, y2=100, color='red', alpha=0.1)
plt.fill_between(daily_df.index, y1=0, y2=30, color='green', alpha=0.1)
plt.title('PNJ Relative Strength Index (RSI)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('RSI', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/rsi.png', dpi=300)
plt.close()

# Plot MACD
print("Creating MACD plot")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['MACD'], label='MACD', color='blue')
plt.plot(daily_df.index, daily_df['MACD_Signal'], label='Signal Line', color='red')
plt.bar(daily_df.index, daily_df['MACD_Histogram'], label='Histogram', color='green', alpha=0.5)
plt.title('PNJ Moving Average Convergence Divergence (MACD)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('MACD', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/macd.png', dpi=300)
plt.close()

# Plot Bollinger Bands
print("Creating Bollinger Bands plot")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['Close'], label='Close Price', color='blue', alpha=0.7)
plt.plot(daily_df.index, daily_df['BB_Middle'], label='20-Day MA', color='orange')
plt.plot(daily_df.index, daily_df['BB_Upper'], label='Upper Band (+2σ)', color='red')
plt.plot(daily_df.index, daily_df['BB_Lower'], label='Lower Band (-2σ)', color='green')
plt.fill_between(daily_df.index, daily_df['BB_Upper'], daily_df['BB_Lower'], color='gray', alpha=0.1)
plt.title('PNJ Bollinger Bands', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/bollinger_bands.png', dpi=300)
plt.close()

# Plot Stochastic Oscillator
print("Creating Stochastic Oscillator plot")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['Stoch_K'], label='%K', color='blue')
plt.plot(daily_df.index, daily_df['Stoch_D'], label='%D', color='red')
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=20, color='green', linestyle='--', alpha=0.5)
plt.fill_between(daily_df.index, y1=80, y2=100, color='red', alpha=0.1)
plt.fill_between(daily_df.index, y1=0, y2=20, color='green', alpha=0.1)
plt.title('PNJ Stochastic Oscillator', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/stochastic.png', dpi=300)
plt.close()

# Plot ADX
print("Creating ADX plot")
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['ADX'], label='ADX', color='black')
plt.plot(daily_df.index, daily_df['DI+'], label='+DI', color='green')
plt.plot(daily_df.index, daily_df['DI-'], label='-DI', color='red')
plt.axhline(y=25, color='blue', linestyle='--', alpha=0.5)
plt.title('PNJ Average Directional Index (ADX)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/adx.png', dpi=300)
plt.close()

# Analyze correlation between indicators and price movements
print("Analyzing correlation between indicators and price movements")
# Calculate next day returns
daily_df['Next_Day_Return'] = daily_df['Close'].pct_change(1).shift(-1)

# Create correlation dataframe
correlation_columns = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                       'Stoch_K', 'Stoch_D', 'ADX', 'DI+', 'DI-']
correlation_df = daily_df[correlation_columns + ['Next_Day_Return']].dropna()

# Calculate correlation
correlation = correlation_df.corr()['Next_Day_Return'].sort_values(ascending=False)

# Plot correlation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Technical Indicators and Price Movements', fontsize=16)
plt.tight_layout()
plt.savefig('stock_analysis/technical_indicators/correlation_heatmap.png', dpi=300)
plt.close()

# Save correlation to file
correlation.to_frame().to_csv('technical_indicators/indicator_correlation.csv')

print("Technical indicator analysis completed. Results saved to stock_analysis/technical_indicators/")
