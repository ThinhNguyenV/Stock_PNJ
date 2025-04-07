import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Create output directory for models
os.makedirs('stock_analysis/models', exist_ok=True)

# Load the data with technical indicators
print("Loading data with technical indicators...")
df = pd.read_csv('stock_analysis/data/pnj_daily_with_indicators.csv')
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df.set_index('Date/Time', inplace=True)

# Display basic information
print("\n--- Basic Information ---")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values and handle them
print("\n--- Checking for missing values ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Drop rows with NaN values (these are mostly at the beginning due to rolling windows)
df_clean = df.dropna()
print(f"\nShape after dropping NaN values: {df_clean.shape}")

# Feature Engineering
print("\n--- Feature Engineering ---")
# Create lag features
print("Creating lag features...")
for i in range(1, 6):
    df_clean[f'Close_Lag_{i}'] = df_clean['Close'].shift(i)
    df_clean[f'Volume_Lag_{i}'] = df_clean['Volume'].shift(i)
    df_clean[f'RSI_Lag_{i}'] = df_clean['RSI'].shift(i)
    df_clean[f'MACD_Lag_{i}'] = df_clean['MACD'].shift(i)

# Create rolling window features
print("Creating rolling window features...")
df_clean['Rolling_Mean_5'] = df_clean['Close'].rolling(window=5).mean().shift(1)
df_clean['Rolling_Mean_10'] = df_clean['Close'].rolling(window=10).mean().shift(1)
df_clean['Rolling_Std_5'] = df_clean['Close'].rolling(window=5).std().shift(1)
df_clean['Rolling_Std_10'] = df_clean['Close'].rolling(window=10).std().shift(1)

# Create target variables for different prediction horizons
print("Creating target variables...")
df_clean['Target_Next_Day'] = df_clean['Close'].shift(-1)
df_clean['Target_Next_Week'] = df_clean['Close'].shift(-5)  # Assuming 5 trading days in a week
df_clean['Target_Next_Month'] = df_clean['Close'].shift(-21)  # Assuming 21 trading days in a month

# Drop rows with NaN values after feature engineering
df_clean = df_clean.dropna()
print(f"Shape after feature engineering: {df_clean.shape}")

# Prepare features and target for next day prediction
print("\n--- Preparing features and target for next day prediction ---")
# Features to use
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
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

# Split the data into training and testing sets (80% train, 20% test)
print("Splitting data into training and testing sets...")
# Use the last 20% of data for testing to maintain time series integrity
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train_next_day, y_test_next_day = y_next_day[:split_idx], y_next_day[split_idx:]
y_train_next_week, y_test_next_week = y_next_week[:split_idx], y_next_week[split_idx:]
y_train_next_month, y_test_next_month = y_next_month[:split_idx], y_next_month[split_idx:]

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Function to plot actual vs predicted values
def plot_predictions(y_true, y_pred, model_name, prediction_horizon):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true.index, y_true.values, label='Actual', color='blue')
    plt.plot(y_true.index, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f'{model_name} - Actual vs Predicted ({prediction_horizon})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'stock_analysis/models/{model_name}_{prediction_horizon}.png', dpi=300)
    plt.close()

# Dictionary to store model results
model_results = {}

# 1. Linear Regression Model
print("\n--- Training Linear Regression Model ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_next_day)
lr_pred = lr_model.predict(X_test)
model_results['Linear Regression'] = evaluate_model(y_test_next_day, lr_pred, "Linear Regression")
plot_predictions(y_test_next_day, lr_pred, "Linear_Regression", "Next_Day")

# 2. Random Forest Model
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_next_day)
rf_pred = rf_model.predict(X_test)
model_results['Random Forest'] = evaluate_model(y_test_next_day, rf_pred, "Random Forest")
plot_predictions(y_test_next_day, rf_pred, "Random_Forest", "Next_Day")

# 3. Gradient Boosting Model
print("\n--- Training Gradient Boosting Model ---")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train_next_day)
gb_pred = gb_model.predict(X_test)
model_results['Gradient Boosting'] = evaluate_model(y_test_next_day, gb_pred, "Gradient Boosting")
plot_predictions(y_test_next_day, gb_pred, "Gradient_Boosting", "Next_Day")

# 4. XGBoost Model
print("\n--- Training XGBoost Model ---")
xgb_model = XGBRegressor.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train_next_day)
xgb_pred = xgb_model.predict(X_test)
model_results['XGBoost'] = evaluate_model(y_test_next_day, xgb_pred, "XGBoost")
plot_predictions(y_test_next_day, xgb_pred, "XGBoost", "Next_Day")

# 5. ARIMA Model (using only the Close price)
print("\n--- Training ARIMA Model ---")
# Use only the Close price for ARIMA
close_series = df_clean['Close']
train_data = close_series[:split_idx]
test_data = close_series[split_idx:]

# Fit ARIMA model
try:
    arima_model = ARIMA(train_data, order=(5,1,0))
    arima_fit = arima_model.fit()
    
    # Make predictions
    arima_pred = arima_fit.forecast(steps=len(test_data))
    model_results['ARIMA'] = evaluate_model(test_data, arima_pred, "ARIMA")
    plot_predictions(test_data, arima_pred, "ARIMA", "Next_Day")
except Exception as e:
    print(f"Error training ARIMA model: {e}")
    model_results['ARIMA'] = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

# Compare model performances
print("\n--- Model Performance Comparison ---")
performance_df = pd.DataFrame(model_results).T
performance_df = performance_df[['rmse', 'mae', 'r2']]
print(performance_df)

# Plot model comparison
plt.figure(figsize=(12, 8))
performance_df['rmse'].plot(kind='bar', color='blue', alpha=0.7)
plt.title('Model Comparison - RMSE (Lower is Better)', fontsize=16)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/models/model_comparison_rmse.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
performance_df['r2'].plot(kind='bar', color='green', alpha=0.7)
plt.title('Model Comparison - R² (Higher is Better)', fontsize=16)
plt.ylabel('R²', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stock_analysis/models/model_comparison_r2.png', dpi=300)
plt.close()

# Feature importance for tree-based models
print("\n--- Feature Importance Analysis ---")
# Random Forest feature importance
plt.figure(figsize=(14, 10))
feature_importance = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Random Forest - Top 20 Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig('stock_analysis/models/rf_feature_importance.png', dpi=300)
plt.close()

# XGBoost feature importance
plt.figure(figsize=(14, 10))
feature_importance = pd.DataFrame({'Feature': features, 'Importance': xgb_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('XGBoost - Top 20 Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig('stock_analysis/models/xgb_feature_importance.png', dpi=300)
plt.close()

# Save the best model
print("\n--- Saving the best model ---")
best_model_name = performance_df['rmse'].idxmin()
print(f"Best model based on RMSE: {best_model_name}")

if best_model_name == 'Linear Regression':
    best_model = lr_model
elif best_model_name == 'Random Forest':
    best_model = rf_model
elif best_model_name == 'Gradient Boosting':
    best_model = gb_model
elif best_model_name == 'XGBoost':
    best_model = xgb_model
else:
    best_model = None

# Save model results to CSV
performance_df.to_csv('stock_analysis/models/model_performance.csv')

print("\nModel building and evaluation completed. Results saved to stock_analysis/models/")
