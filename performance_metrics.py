import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
import os

# Directory paths
FORECAST_DIR = "analysis/forecasts/"
METRICS_CSV = "analysis/performance_metrics.csv"
METRICS_JSON = "analysis/performance_metrics.json"

# Function to calculate performance metrics
def calculate_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {"MSE": mse, "MAE": mae, "R2_Score": r2}

# Function to evaluate all stocks and models
def evaluate_forecasts():
    metrics_list = []
    combined_metrics = {}

    # Fetch all forecast CSV files
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, "*.csv"))

    for file_path in forecast_files:
        # Extract the stock symbol and model name from the file name
        filename = os.path.basename(file_path)
        symbol, model = filename.replace("forecast_", "").replace(".csv", "").split("_", 1)

        try:
            # Load the forecasted data
            forecast_df = pd.read_csv(file_path)
            
            # Check if the required columns are present
            if 'Date' not in forecast_df.columns or model + '_Forecast' not in forecast_df.columns:
                print(f"Skipping {filename}: Missing required columns.")
                continue

            # Load true values from the database (assuming 'stock_data' table in PostgreSQL)
            true_df = pd.read_sql(f"SELECT date, close FROM stock_data WHERE stock_symbol = '{symbol}' ORDER BY date", db_engine)
            true_df = true_df.tail(len(forecast_df))

            # Align true and forecasted data
            true_values = true_df['close'].values
            predicted_values = forecast_df[model + '_Forecast'].values

            # Calculate metrics
            metrics = calculate_metrics(true_values, predicted_values)
            metrics['Stock'] = symbol
            metrics['Model'] = model

            # Append metrics to list
            metrics_list.append(metrics)

            # Add to combined JSON data
            if symbol not in combined_metrics:
                combined_metrics[symbol] = {}
            combined_metrics[symbol][model] = metrics

            print(f"Metrics calculated for {symbol} using {model} model.")
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"Performance metrics saved to {METRICS_CSV}")

    # Save metrics to JSON
    with open(METRICS_JSON, 'w') as json_file:
        json.dump(combined_metrics, json_file, indent=4)
    print(f"Combined performance metrics saved to {METRICS_JSON}")

    # Generate visualizations
    visualize_metrics(metrics_df)

# Function to visualize metrics using Plotly
def visualize_metrics(metrics_df):
    try:
        # Visualization of MSE for all models
        fig = px.bar(metrics_df, x='Stock', y='MSE', color='Model', title='MSE Comparison Across Models')
        fig.update_layout(xaxis_title='Stock', yaxis_title='Mean Squared Error')
        fig.show()

        # Visualization of MAE for all models
        fig = px.bar(metrics_df, x='Stock', y='MAE', color='Model', title='MAE Comparison Across Models')
        fig.update_layout(xaxis_title='Stock', yaxis_title='Mean Absolute Error')
        fig.show()

        # Visualization of R2 Score for all models
        fig = go.Figure()
        for model in metrics_df['Model'].unique():
            model_data = metrics_df[metrics_df['Model'] == model]
            fig.add_trace(go.Scatter(x=model_data['Stock'], y=model_data['R2_Score'],
                                     mode='lines+markers', name=model))

        fig.update_layout(title='R2 Score Comparison Across Models',
                          xaxis_title='Stock',
                          yaxis_title='R2 Score')
        fig.show()

        print("Visualizations generated successfully.")
    except Exception as e:
        print(f"Error in visualization: {e}")

# Run the evaluation script
if __name__ == "__main__":
    evaluate_forecasts()