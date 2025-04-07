import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os
from scipy import stats
import matplotlib.dates as mdates

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Create output directory for predictions
os.makedirs('stock_analysis/predictions', exist_ok=True)

# Load the data with technical indicators
print("Loading data with technical indicators...")
df = pd.read_csv('stock_analysis/data/pnj_daily_with_indicators.csv')
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df.set_index('Date/Time', inplace=True)

# Drop rows with NaN values
df_clean = df.dropna()

# Feature Engineering (same as in build_model.py)
print("\nPerforming feature engineering...")
# Create lag features
for i in range(1, 6):
    df_clean[f'Close_Lag_{i}'] = df_clean['Close'].shift(i)
    df_clean[f'Volume_Lag_{i}'] = df_clean['Volume'].shift(i)
    df_clean[f'RSI_Lag_{i}'] = df_clean['RSI'].shift(i)
    df_clean[f'MACD_Lag_{i}'] = df_clean['MACD'].shift(i)

# Create rolling window features
df_clean['Rolling_Mean_5'] = df_clean['Close'].rolling(window=5).mean().shift(1)
df_clean['Rolling_Mean_10'] = df_clean['Close'].rolling(window=10).mean().shift(1)
df_clean['Rolling_Std_5'] = df_clean['Close'].rolling(window=5).std().shift(1)
df_clean['Rolling_Std_10'] = df_clean['Close'].rolling(window=10).std().shift(1)

# Create target variables
df_clean['Target_Next_Day'] = df_clean['Close'].shift(-1)
df_clean['Target_Next_Week'] = df_clean['Close'].shift(-5)  # Assuming 5 trading days in a week
df_clean['Target_Next_Month'] = df_clean['Close'].shift(-21)  # Assuming 21 trading days in a month

# Drop rows with NaN values after feature engineering
df_clean = df_clean.dropna()

# Features to use (same as in build_model.py)
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
            'BB_Middle', 'BB_Upper', 'BB_Lower', 
            'Stoch_K', 'Stoch_D', 'ADX', 'DI+', 'DI-',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_4', 'Volume_Lag_5',
            'RSI_Lag_1', 'RSI_Lag_2', 'RSI_Lag_3', 'RSI_Lag_4', 'RSI_Lag_5',
            'MACD_Lag_1', 'MACD_Lag_2', 'MACD_Lag_3', 'MACD_Lag_4', 'MACD_Lag_5',
            'Rolling_Mean_5', 'Rolling_Mean_10', 'Rolling_Std_5', 'Rolling_Std_10']

X = df_clean[features]
y_next_day = df_clean['Target_Next_Day']
y_next_week = df_clean['Target_Next_Week']
y_next_month = df_clean['Target_Next_Month']

# Scale the features
print("Scaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train the Linear Regression model on the full dataset
print("\nTraining Linear Regression model on full dataset...")
lr_model = LinearRegression()
lr_model.fit(X_scaled, y_next_day)

# Function to generate predictions for future dates
def generate_future_predictions(model, last_data, num_days=30):
    print(f"\nGenerating predictions for next {num_days} days...")
    
    # Create a dataframe to store predictions
    future_dates = [last_data.index[-1] + timedelta(days=i+1) for i in range(num_days)]
    predictions_df = pd.DataFrame(index=future_dates, columns=['Predicted_Close', 'Lower_CI', 'Upper_CI'])
    
    # Get the last available data point
    current_data = last_data.iloc[-1:].copy()
    
    # Generate predictions for each future day
    for i in range(num_days):
        # Prepare the features for prediction
        if i == 0:
            # For the first prediction, use the last available data
            pred_features = current_data[features].values
        else:
            # For subsequent predictions, update the features based on previous predictions
            # Update Close and lag features
            current_data['Close'] = predictions_df['Predicted_Close'].iloc[i-1]
            
            for j in range(5, 1, -1):
                current_data[f'Close_Lag_{j}'] = current_data[f'Close_Lag_{j-1}']
            current_data['Close_Lag_1'] = current_data['Close']
            
            # Update other features (simplified approach)
            # In a real scenario, you would need to update all features based on their definitions
            current_data['Rolling_Mean_5'] = (current_data['Close'] + 
                                             current_data['Close_Lag_1'] + 
                                             current_data['Close_Lag_2'] + 
                                             current_data['Close_Lag_3'] + 
                                             current_data['Close_Lag_4']) / 5
            
            current_data['Rolling_Mean_10'] = current_data['Rolling_Mean_5']  # Simplified
            current_data['Rolling_Std_5'] = current_data['Rolling_Std_5']  # Keep the same
            current_data['Rolling_Std_10'] = current_data['Rolling_Std_10']  # Keep the same
            
            pred_features = current_data[features].values
        
        # Scale the features
        pred_features_scaled = scaler.transform(pred_features)
        
        # Make prediction
        prediction = model.predict(pred_features_scaled)[0]
        
        # Calculate confidence intervals (using the RMSE from evaluation)
        rmse = 1.28  # From model evaluation
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * rmse
        
        lower_ci = prediction - margin_of_error
        upper_ci = prediction + margin_of_error
        
        # Store the prediction and confidence intervals
        predictions_df.loc[future_dates[i], 'Predicted_Close'] = prediction
        predictions_df.loc[future_dates[i], 'Lower_CI'] = lower_ci
        predictions_df.loc[future_dates[i], 'Upper_CI'] = upper_ci
    
    return predictions_df

# Generate predictions for the next 30 days
future_predictions = generate_future_predictions(lr_model, df_clean, num_days=30)

# Print the predictions
print("\nPredictions for the next 30 days:")
print(future_predictions)

# Plot the predictions without using fill_between
print("\nPlotting predictions...")
plt.figure(figsize=(14, 7))

# Plot historical data (last 60 days)
historical_data = df_clean['Close'].iloc[-60:]
plt.plot(historical_data.index, historical_data, label='Historical Close Price', color='blue')

# Plot predictions
plt.plot(future_predictions.index, future_predictions['Predicted_Close'], label='Predicted Close Price', color='red', linestyle='--')

# Plot upper and lower confidence intervals as separate lines
plt.plot(future_predictions.index, future_predictions['Upper_CI'], color='red', alpha=0.3, linestyle=':')
plt.plot(future_predictions.index, future_predictions['Lower_CI'], color='red', alpha=0.3, linestyle=':')

# Add labels and title
plt.title('PNJ Stock Price Prediction for Next 30 Days', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('stock_analysis/predictions/future_predictions.png', dpi=300)
plt.close()

# Generate buy/sell signals based on predictions
print("\nGenerating buy/sell signals...")
# Calculate the predicted price change
future_predictions['Price_Change'] = future_predictions['Predicted_Close'].diff()
future_predictions['Price_Change_Pct'] = future_predictions['Predicted_Close'].pct_change() * 100

# Define thresholds for buy/sell signals
buy_threshold = 1.0  # 1% increase
sell_threshold = -1.0  # 1% decrease

# Generate signals
future_predictions['Signal'] = 'Hold'
future_predictions.loc[future_predictions['Price_Change_Pct'] >= buy_threshold, 'Signal'] = 'Buy'
future_predictions.loc[future_predictions['Price_Change_Pct'] <= sell_threshold, 'Signal'] = 'Sell'

# Count the number of signals
buy_signals = (future_predictions['Signal'] == 'Buy').sum()
sell_signals = (future_predictions['Signal'] == 'Sell').sum()
hold_signals = (future_predictions['Signal'] == 'Hold').sum()

print(f"Buy signals: {buy_signals}")
print(f"Sell signals: {sell_signals}")
print(f"Hold signals: {hold_signals}")

# Plot the signals
plt.figure(figsize=(14, 7))

# Plot predictions
plt.plot(future_predictions.index, future_predictions['Predicted_Close'], label='Predicted Close Price', color='blue')

# Plot buy signals
buy_points = future_predictions[future_predictions['Signal'] == 'Buy']
if not buy_points.empty:
    plt.scatter(buy_points.index, buy_points['Predicted_Close'], color='green', s=100, marker='^', label='Buy Signal')

# Plot sell signals
sell_points = future_predictions[future_predictions['Signal'] == 'Sell']
if not sell_points.empty:
    plt.scatter(sell_points.index, sell_points['Predicted_Close'], color='red', s=100, marker='v', label='Sell Signal')

# Add labels and title
plt.title('PNJ Stock Price Prediction with Buy/Sell Signals', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('stock_analysis/predictions/buy_sell_signals.png', dpi=300)
plt.close()

# Save the predictions to CSV
future_predictions.to_csv('stock_analysis/predictions/future_predictions.csv')

# Create a summary of predictions
print("\nCreating prediction summary...")
prediction_summary = pd.DataFrame({
    'Metric': ['Last Close Price', 'Predicted Price (Next Day)', 'Predicted Price (30 Days)', 
              'Expected Change (30 Days)', 'Expected Change % (30 Days)',
              'Buy Signals', 'Sell Signals', 'Hold Signals'],
    'Value': [df_clean['Close'].iloc[-1],
             future_predictions['Predicted_Close'].iloc[0],
             future_predictions['Predicted_Close'].iloc[-1],
             future_predictions['Predicted_Close'].iloc[-1] - df_clean['Close'].iloc[-1],
             ((future_predictions['Predicted_Close'].iloc[-1] / df_clean['Close'].iloc[-1]) - 1) * 100,
             buy_signals, sell_signals, hold_signals]
})

# Save the summary to CSV
prediction_summary.to_csv('stock_analysis/predictions/prediction_summary.csv', index=False)
print(prediction_summary)

# Create a text file with prediction insights
print("\nCreating prediction insights document...")
with open('stock_analysis/predictions/prediction_insights.txt', 'w') as f:
    f.write("# PNJ Stock Price Prediction Insights\n\n")
    
    f.write("## Prediction Summary\n\n")
    f.write(f"Last Close Price: {df_clean['Close'].iloc[-1]:.2f}\n")
    f.write(f"Predicted Price (Next Day): {future_predictions['Predicted_Close'].iloc[0]:.2f}\n")
    f.write(f"Predicted Price (30 Days): {future_predictions['Predicted_Close'].iloc[-1]:.2f}\n")
    f.write(f"Expected Change (30 Days): {future_predictions['Predicted_Close'].iloc[-1] - df_clean['Close'].iloc[-1]:.2f}\n")
    f.write(f"Expected Change % (30 Days): {((future_predictions['Predicted_Close'].iloc[-1] / df_clean['Close'].iloc[-1]) - 1) * 100:.2f}%\n\n")
    
    f.write("## Buy/Sell Signals\n\n")
    f.write(f"Buy Signals: {buy_signals}\n")
    f.write(f"Sell Signals: {sell_signals}\n")
    f.write(f"Hold Signals: {hold_signals}\n\n")
    
    f.write("## Prediction Confidence\n\n")
    f.write(f"The predictions are made with a 95% confidence interval, with an average margin of error of ±{1.28 * 1.96:.2f} based on the model's RMSE.\n\n")
    
    f.write("## Investment Recommendations\n\n")
    
    # Determine overall recommendation based on predictions
    avg_change = future_predictions['Price_Change_Pct'].mean()
    if avg_change > 1.0:
        recommendation = "BUY"
        reason = f"The model predicts an average daily price increase of {avg_change:.2f}% over the next 30 days, suggesting a positive trend."
    elif avg_change < -1.0:
        recommendation = "SELL"
        reason = f"The model predicts an average daily price decrease of {abs(avg_change):.2f}% over the next 30 days, suggesting a negative trend."
    else:
        recommendation = "HOLD"
        reason = f"The model predicts relatively stable prices with an average daily change of {avg_change:.2f}% over the next 30 days."
    
    f.write(f"Overall Recommendation: {recommendation}\n")
    f.write(f"Reason: {reason}\n\n")
    
    f.write("## Important Notes\n\n")
    f.write("1. These predictions are based on historical data and technical indicators only.\n")
    f.write("2. External factors such as market news, economic events, and company announcements are not considered in this model.\n")
    f.write("3. The stock market is inherently unpredictable, and all investments carry risk.\n")
    f.write("4. This analysis should be used as one of many tools for making investment decisions, not as the sole basis.\n")
    f.write("5. Past performance is not indicative of future results.\n")

print("\nStock prediction completed. Results saved to stock_analysis/predictions/")
