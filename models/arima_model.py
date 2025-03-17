import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import logging
import traceback
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

def preprocess_data(stock_data):
    """
    Preprocess input data for ARIMA modeling.
    
    Args:
        stock_data (pd.DataFrame or np.ndarray): Input stock data
    
    Returns:
        tuple: Processed prices and scaler
    """
    try:
        # Handle different input types
        if isinstance(stock_data, pd.DataFrame):
            # Check for correct column
            if 'Close' in stock_data.columns:
                prices = stock_data['Close'].values
            elif 'y' in stock_data.columns:
                prices = stock_data['y'].values
            elif stock_data.shape[1] == 1:
                # Assume first column is the price
                prices = stock_data.iloc[:, 0].values
            else:
                logger.error(f"No valid price column found. Columns: {stock_data.columns.tolist()}")
                return None, None
        elif isinstance(stock_data, np.ndarray):
            # If it's already a NumPy array, use it directly
            prices = stock_data.flatten()
        elif isinstance(stock_data, pd.Series):
            prices = stock_data.values
        else:
            logger.error(f"Unsupported input type: {type(stock_data)}")
            return None, None

        # Ensure prices is a NumPy array
        prices = np.asarray(prices, dtype=float)
        
        # Scale the data
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        return scaled_prices, scaler
    
    except Exception as e:
        logger.error(f"Data preprocessing error: {e}")
        logger.error(traceback.format_exc())
        return None, None

def arima_model(stock_data, symbol, db_engine, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the stock data and generate forecasts.
    
    Args:
        stock_data (pd.DataFrame or np.ndarray): Input stock data
        symbol (str): Stock symbol
        db_engine: SQLAlchemy database engine
        order (tuple): ARIMA model order (p,d,q)
    
    Returns:
        pd.DataFrame: Forecast results or None if error
    """
    try:
        # Preprocess data
        scaled_prices, scaler = preprocess_data(stock_data)
        
        # Validate preprocessed data
        if scaled_prices is None or scaler is None:
            logger.error(f"Failed to preprocess data for {symbol}")
            return None

        # Check for sufficient data
        if len(scaled_prices) < 30:
            logger.warning(f"Insufficient data points for {symbol}: {len(scaled_prices)}")
            return None

        # Fit ARIMA model
        try:
            model = ARIMA(scaled_prices, order=order)
            model_fit = model.fit()
        except Exception as model_err:
            logger.error(f"Model fitting error for {symbol}: {model_err}")
            logger.error(traceback.format_exc())
            return None

        # Generate forecast
        try:
            forecast_scaled = model_fit.forecast(steps=30)
            
            # Inverse transform forecast
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        except Exception as forecast_err:
            logger.error(f"Forecast generation error for {symbol}: {forecast_err}")
            logger.error(traceback.format_exc())
            return None

        # Create forecast dates
        last_date = (stock_data.index[-1] if isinstance(stock_data, pd.DataFrame) 
                     else pd.Timestamp.now())
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=30,
            freq='B'
        )

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_close': forecast,
            'stock_symbol': symbol
        })

        # Save forecasts to database
        try:
            with db_engine.connect() as connection:
                # Prepare SQL insert statement
                insert_query = text("""
                    INSERT INTO stock_forecasts_arima (date, predicted_close, stock_symbol)
                    VALUES (:date, :predicted_close, :stock_symbol)
                """)
                
                # Execute batch insert
                connection.execute(
                    insert_query, 
                    forecast_df.to_dict('records')
                )
                connection.commit()
                
                logger.info(f"ARIMA forecasts saved to database for {symbol}")
        
        except SQLAlchemyError as db_error:
            logger.error(f"Database error saving ARIMA forecasts for {symbol}: {db_error}")
            logger.error(traceback.format_exc())
        
        # Rename columns for return DataFrame
        forecast_df = forecast_df.rename(columns={
            'date': 'Date', 
            'predicted_close': 'Predicted_Close'
        })

        logger.info(f"ARIMA forecast completed for {symbol}")
        return forecast_df

    except Exception as e:
        logger.error(f"ARIMA model error for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return None