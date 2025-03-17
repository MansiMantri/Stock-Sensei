import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(stock_data, arima_forecast, sarima_forecast, prophet_forecast, svm_forecast, rnn_forecast, symbol):
    # Create directories for saving visualizations
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # Convert 'date' column to datetime
    if 'date' in stock_data.columns:
        stock_data['date'] = pd.to_datetime(stock_data['date'])
    else:
        print(f"Warning: 'date' column not found in stock data for symbol {symbol}")

    if 'date' in arima_forecast.columns:
        arima_forecast['date'] = pd.to_datetime(arima_forecast['date'])
    else:
        print(f"Warning: 'date' column not found in ARIMA forecast for symbol {symbol}")

    if 'date' in sarima_forecast.columns:
        sarima_forecast['date'] = pd.to_datetime(sarima_forecast['date'])
    else:
        print(f"Warning: 'date' column not found in SARIMA forecast for symbol {symbol}")

    if 'date' in prophet_forecast.columns:
        prophet_forecast['date'] = pd.to_datetime(prophet_forecast['date'])
    else:
        print(f"Warning: 'date' column not found in Prophet forecast for symbol {symbol}")

    if 'date' in svm_forecast.columns:
        svm_forecast['date'] = pd.to_datetime(svm_forecast['date'])
    else:
        print(f"Warning: 'date' column not found in SVM forecast for symbol {symbol}")

    if 'date' in rnn_forecast.columns:
        rnn_forecast['date'] = pd.to_datetime(rnn_forecast['date'])
    else:
        print(f"Warning: 'date' column not found in RNN forecast for symbol {symbol}")

    # Plot the actual and forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['date'], stock_data['close'], label='Actual')
    plt.plot(arima_forecast['date'], arima_forecast['forecast'], label='ARIMA')
    plt.plot(sarima_forecast['date'], sarima_forecast['forecast'], label='SARIMA')
    plt.plot(prophet_forecast['date'], prophet_forecast['forecast'], label='Prophet')
    plt.plot(svm_forecast['date'], svm_forecast['forecast'], label='SVM')
    plt.plot(rnn_forecast['date'], rnn_forecast['forecast'], label='RNN')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f"{symbol} - Stock Price Forecasting")
    plt.legend()
    plt.savefig(os.path.join(vis_dir, f"{symbol}_forecasting.png"))
    plt.close()

    # Plot the forecast errors
    plt.figure(figsize=(12, 6))
    plt.plot(arima_forecast['date'], arima_forecast['error'], label='ARIMA')
    plt.plot(sarima_forecast['date'], sarima_forecast['error'], label='SARIMA')
    plt.plot(prophet_forecast['date'], prophet_forecast['error'], label='Prophet')
    plt.plot(svm_forecast['date'], svm_forecast['error'], label='SVM')
    plt.plot(rnn_forecast['date'], rnn_forecast['error'], label='RNN')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.title(f"{symbol} - Forecast Errors")
    plt.legend()
    plt.savefig(os.path.join(vis_dir, f"{symbol}_forecast_errors.png"))
    plt.close()