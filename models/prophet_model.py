import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

def prophet_model(stock_data, symbol, db_engine):
    """
    Fit a Prophet model to the stock data and generate forecasts.
    
    Args:
        stock_data (pd.DataFrame): Input stock data with 'ds' and 'y' columns
        symbol (str): Stock symbol
        db_engine: SQLAlchemy database engine
    
    Returns:
        pd.DataFrame: Forecast results or None if error
    """
    try:
        # Validate input data
        if not isinstance(stock_data, pd.DataFrame):
            logger.error(f"Invalid input type for {symbol}: {type(stock_data)}")
            return None

        # Validate required columns
        required_columns = {'ds', 'y'}
        if not all(col in stock_data.columns for col in required_columns):
            logger.error(f"Missing required columns for {symbol}. Required: {required_columns}, Found: {stock_data.columns.tolist()}")
            return None

        # Check for sufficient data
        if len(stock_data) < 30:
            logger.warning(f"Insufficient data points for {symbol}: {len(stock_data)}")
            return None

        # Prepare Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        # Fit model
        model.fit(stock_data)

        # Create future dates for forecasting
        future = model.make_future_dataframe(periods=30, freq='B')
        
        # Generate forecast
        forecast = model.predict(future)

        # Extract relevant forecast data
        forecast_df = pd.DataFrame({
            'date': forecast['ds'].tail(30),
            'predicted_close': forecast['yhat'].tail(30),
            'stock_symbol': symbol
        })

        # Save forecasts to database
        try:
            with db_engine.connect() as connection:
                # Prepare SQL insert statement
                insert_query = text("""
                    INSERT INTO stock_forecasts_prophet (date, predicted_close, stock_symbol)
                    VALUES (:date, :predicted_close, :stock_symbol)
                """)
                
                # Execute batch insert
                connection.execute(
                    insert_query, 
                    forecast_df.to_dict('records')
                )
                connection.commit()
                
                logger.info(f"Prophet forecasts saved to database for {symbol}")
        
        except SQLAlchemyError as db_error:
            logger.error(f"Database error saving Prophet forecasts for {symbol}: {db_error}")
        
        # Rename columns for return DataFrame
        forecast_df = forecast_df.rename(columns={
            'date': 'Date', 
            'predicted_close': 'Predicted_Close'
        })

        logger.info(f"Prophet forecast completed for {symbol}")
        return forecast_df

    except Exception as e:
        logger.error(f"Prophet model error for {symbol}: {e}")
        return None