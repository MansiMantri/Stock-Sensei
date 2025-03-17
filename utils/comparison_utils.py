import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score
)
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)

def calculate_metrics(actual, predicted):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        actual (np.array or pd.Series): Actual values
        predicted (np.array or pd.Series): Predicted values
    
    Returns:
        dict: Dictionary of performance metrics
    """
    try:
        # Convert to numpy arrays
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        
        # Ensure matching lengths
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Remove any infinite or NaN values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        # Check if we have sufficient data
        if len(actual) < 2:
            logger.warning("Insufficient data for metric calculation")
            return {}
        
        # Calculate metrics
        metrics = {}
        
        # Mean Absolute Error
        try:
            metrics['MAE'] = mean_absolute_error(actual, predicted)
        except Exception as mae_error:
            logger.error(f"Error calculating MAE: {mae_error}")
            metrics['MAE'] = np.nan
        
        # Mean Squared Error
        try:
            metrics['MSE'] = mean_squared_error(actual, predicted)
        except Exception as mse_error:
            logger.error(f"Error calculating MSE: {mse_error}")
            metrics['MSE'] = np.nan
        
        # Root Mean Squared Error
        try:
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
        except Exception as rmse_error:
            logger.error(f"Error calculating RMSE: {rmse_error}")
            metrics['RMSE'] = np.nan
        
        # R-squared
        try:
            metrics['R2'] = r2_score(actual, predicted)
        except Exception as r2_error:
            logger.error(f"Error calculating R2: {r2_error}")
            metrics['R2'] = np.nan
        
        # Mean Absolute Percentage Error (MAPE)
        try:
            # Handle zero division
            mape_mask = (actual != 0)
            if np.any(mape_mask):
                mape = np.mean(np.abs((actual[mape_mask] - predicted[mape_mask]) / actual[mape_mask])) * 100
                metrics['MAPE'] = mape
            else:
                metrics['MAPE'] = np.nan
        except Exception as mape_error:
            logger.error(f"Error calculating MAPE: {mape_error}")
            metrics['MAPE'] = np.nan
        
        return metrics
    
    except Exception as e:
        logger.error(f"Unexpected error in metric calculation: {e}")
        logger.error(traceback.format_exc())
        return {}

def compare_models(stock_data, forecasts):
    """
    Compare performance metrics for different forecasting models.
    
    Args:
        stock_data (pd.DataFrame): Actual stock price data
        forecasts (dict): Dictionary of forecasts from different models
    
    Returns:
        pd.DataFrame: Comparative metrics for different models
    """
    try:
        # Ensure we have actual values
        actual = stock_data['Close'].values[-30:]
        
        # Prepare metrics dictionary
        model_metrics = {}
        
        # Iterate through available forecasts
        for model_name, forecast in forecasts.items():
            try:
                # Extract predicted values
                predicted = forecast['Predicted_Close'].values[:30]
                
                # Ensure matching lengths
                min_length = min(len(actual), len(predicted))
                actual_subset = actual[:min_length]
                predicted_subset = predicted[:min_length]
                
                # Calculate metrics for this model
                metrics = calculate_metrics(actual_subset, predicted_subset)
                
                # Add model name to metrics
                metrics['Model'] = model_name
                
                # Store metrics
                model_metrics[model_name] = metrics
            
            except Exception as model_error:
                logger.error(f"Error processing {model_name} forecast: {model_error}")
                logger.error(traceback.format_exc())
                continue
        
        # Convert to DataFrame
        if model_metrics:
            metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
            
            # Reorder columns to ensure consistent structure
            column_order = ['Model', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE']
            metrics_df = metrics_df.reindex(columns=column_order)
            
            return metrics_df
        else:
            logger.warning("No valid metrics calculated")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def rank_models(metrics_df):
    """
    Rank models based on different performance metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with model performance metrics
    
    Returns:
        dict: Ranking of models for different metrics
    """
    try:
        # Metrics to consider for ranking
        ranking_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        
        # Initialize ranking dictionary
        model_rankings = {}
        
        # Rank models for each metric
        for metric in ranking_metrics:
            # Lower is better for error metrics
            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                ranked_models = metrics_df.sort_values(metric).reset_index()
                ranked_models['Rank'] = ranked_models.index + 1
            # Higher is better for R2
            else:
                ranked_models = metrics_df.sort_values(metric, ascending=False).reset_index()
                ranked_models['Rank'] = ranked_models.index + 1
            
            model_rankings[metric] = dict(zip(ranked_models['Model'], ranked_models['Rank']))
        
        return model_rankings
    
    except Exception as e:
        logger.error(f"Error ranking models: {e}")
        logger.error(traceback.format_exc())
        return {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)