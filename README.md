# PNJ Stock Analysis and Prediction Report

## Executive Summary

This report presents a comprehensive analysis of PNJ stock data from February 2018 to December 2020. The analysis includes exploratory data analysis, visualization of stock trends, technical indicator analysis, and predictive modeling. The Linear Regression model was identified as the best-performing model with an RMSE of 1.28 and R² of 0.96. The model predicts a slight decrease in PNJ stock price over the next 30 days, with an expected change of -1.44%. Based on these predictions, the overall investment recommendation is to HOLD.

## 1. Introduction

### 1.1 Dataset Overview

The dataset contains minute-by-minute trading data for PNJ stock spanning from February 28, 2018, to December 22, 2020, with a total of 125,310 records. The data includes the following columns:
- Ticker: Stock symbol (PNJ)
- Date/Time: Trading timestamp
- Open: Opening price
- High: Highest price during the period
- Low: Lowest price during the period
- Close: Closing price
- Volume: Number of shares traded
- Open Interest: Number of open contracts

### 1.2 Analysis Approach

The analysis followed a structured approach:
1. Exploratory Data Analysis: Examining data structure, checking for missing values, and calculating basic statistics
2. Data Visualization: Creating price charts, volume analysis, and identifying trends
3. Technical Indicator Analysis: Calculating and analyzing RSI, MACD, Bollinger Bands, and other indicators
4. Predictive Modeling: Building and evaluating multiple models to predict future stock prices
5. Stock Prediction: Generating predictions for the next 30 days with confidence intervals

## 2. Exploratory Data Analysis

### 2.1 Data Structure and Quality

The dataset contains 125,309 rows of minute-by-minute trading data spanning 709 trading days. There were no missing values in the original dataset. The data was resampled to daily, weekly, and monthly timeframes for different levels of analysis.

### 2.2 Basic Statistics

Key statistics for the PNJ stock:
- Average closing price: 72.18
- Standard deviation: 9.90
- Minimum price: 44.14
- Maximum price: 100.16
- Average daily trading volume: 3,032 shares

### 2.3 Price Distribution and Volatility

The stock showed an annualized volatility of 36.89%, indicating moderate to high price fluctuations. The daily returns distribution was approximately normal with a slight positive skew, suggesting occasional large positive returns.

## 3. Stock Trend Visualization

### 3.1 Price Trends

The stock price showed significant fluctuations over the analyzed period. There was a sharp decline in early 2020 (likely due to the COVID-19 pandemic), followed by a recovery in the latter part of 2020. The 200-day moving average showed an overall upward trend in the latter half of 2020.

### 3.2 Volume Analysis

Trading volume was highly variable, with occasional spikes indicating periods of high market interest. Higher trading volumes were often associated with significant price movements, particularly during market downturns.

### 3.3 Seasonal Patterns

The monthly returns heatmap revealed some seasonal patterns, with certain months consistently showing better performance than others. However, these patterns were not strong enough to form the basis of a trading strategy on their own.

## 4. Technical Indicator Analysis

### 4.1 Relative Strength Index (RSI)

The RSI analysis showed that PNJ stock occasionally entered overbought (RSI > 70) and oversold (RSI < 30) territories, providing potential trading signals. However, the correlation between RSI and next-day returns was relatively weak (-0.021).

### 4.2 Moving Average Convergence Divergence (MACD)

The MACD analysis revealed several crossover signals throughout the analyzed period. The MACD Histogram showed the highest positive correlation with next-day returns (0.055) among all technical indicators, suggesting it may have some predictive value.

### 4.3 Bollinger Bands

Bollinger Bands analysis showed that the stock price occasionally touched or exceeded the upper and lower bands, indicating potential reversal points. The stock generally respected these bands, returning to the middle band after touching the extremes.

### 4.4 Indicator Correlation Analysis

The correlation analysis between technical indicators and next-day returns showed:
- MACD Histogram: 0.055 (highest positive correlation)
- ADX: 0.024
- Stochastic D: -0.002
- DI-: -0.007
- RSI: -0.021
- Stochastic K: -0.028
- DI+: -0.030
- MACD: -0.036
- MACD Signal: -0.058 (highest negative correlation)

These relatively weak correlations suggest that no single indicator is strongly predictive on its own, highlighting the need for a multi-factor approach.

## 5. Predictive Modeling

### 5.1 Model Development

Several models were developed to predict the next day's closing price:
1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. ARIMA

The models used various features including price data, volume, technical indicators, and lag features. The data was split into training (80%) and testing (20%) sets, with the most recent data used for testing.

### 5.2 Feature Engineering

The following features were engineered for the models:
- Lag features for Close price, Volume, RSI, and MACD (1-5 days)
- Rolling mean and standard deviation (5 and 10 days)
- Technical indicators (RSI, MACD, Bollinger Bands, Stochastic Oscillator, ADX)

### 5.3 Model Performance Comparison

Model performance metrics on the test set:

| Model | RMSE | MAE | R² |
|-------|------|-----|---|
| Linear Regression | 1.28 | 0.97 | 0.96 |
| Random Forest | 2.42 | 1.89 | 0.86 |
| Gradient Boosting | 2.23 | 1.71 | 0.88 |
| XGBoost | 2.24 | 1.77 | 0.88 |
| ARIMA | 6.30 | 4.74 | -0.03 |

The Linear Regression model significantly outperformed all other models with the lowest RMSE (1.28) and highest R² (0.96). The ARIMA model performed poorly with a negative R² value, indicating it performs worse than a simple mean-based prediction.

### 5.4 Feature Importance

The feature importance analysis from the tree-based models revealed that recent closing prices, moving averages, and certain technical indicators (particularly RSI and MACD) were the most important features for prediction.

## 6. Stock Predictions

### 6.1 Prediction Methodology

The Linear Regression model (best performer) was used to generate predictions for the next 30 days. The predictions included:
- Point forecasts for closing prices
- 95% confidence intervals based on the model's RMSE
- Buy/sell signals based on predicted price changes

### 6.2 Prediction Results

Key prediction results:
- Last Close Price: 76.70
- Predicted Price (Next Day): 76.43
- Predicted Price (30 Days): 75.60
- Expected Change (30 Days): -1.10 (-1.44%)
- Buy Signals: 0
- Sell Signals: 0
- Hold Signals: 30

The model predicts a slight decrease in PNJ stock price over the next 30 days, with relatively stable day-to-day movements (no significant buy or sell signals based on a 1% threshold).

### 6.3 Confidence Intervals

The predictions are made with a 95% confidence interval, with an average margin of error of ±2.51 based on the model's RMSE. This means we can be 95% confident that the actual price will fall within this range, assuming the model's error distribution remains consistent.

## 7. Investment Recommendations

### 7.1 Overall Recommendation

Based on the predictive model and analysis, the overall recommendation is to **HOLD** PNJ stock. The model predicts relatively stable prices with a slight downward trend over the next 30 days.

### 7.2 Rationale

The recommendation is based on:
1. The predicted price decrease is relatively small (-1.44% over 30 days)
2. No strong buy or sell signals were identified
3. The stock has shown resilience and recovery after previous downturns
4. Technical indicators suggest a neutral market sentiment

### 7.3 Risk Assessment

Key risks to consider:
- The model's predictions have a margin of error (±2.51 at 95% confidence)
- External factors not captured in the model (e.g., market news, economic events) could significantly impact the stock price
- The stock has shown high volatility in the past (36.89% annualized)

## 8. Limitations and Future Work

### 8.1 Limitations

This analysis has several limitations:
1. It relies solely on historical price and volume data, without incorporating fundamental analysis
2. External factors such as market news, economic events, and company announcements are not considered
3. The stock market is inherently unpredictable, and all investments carry risk
4. Past performance is not indicative of future results

### 8.2 Future Work

Potential improvements for future analysis:
1. Incorporate fundamental analysis (e.g., financial statements, valuation metrics)
2. Include external data sources (e.g., market indices, economic indicators, news sentiment)
3. Explore deep learning models like LSTM which may capture more complex patterns
4. Develop ensemble methods that combine multiple models
5. Implement a more sophisticated trading strategy with dynamic thresholds

## 9. Conclusion

This comprehensive analysis of PNJ stock has provided valuable insights into its historical performance, technical indicators, and potential future movements. The Linear Regression model demonstrated strong predictive performance and suggests a relatively stable price with a slight downward trend over the next 30 days. Based on these findings, the recommendation is to HOLD PNJ stock while monitoring for any significant changes in market conditions or company fundamentals.

The analysis should be used as one of many tools for making investment decisions, not as the sole basis. Investors should consider their own risk tolerance, investment horizon, and overall portfolio strategy when making investment decisions.
