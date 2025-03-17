import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_db_engine(db_url=None):
    """
    Create and return a SQLAlchemy database engine.

    Args:
        db_url (str, optional): Database connection URL. Defaults to None.

    Returns:
        sqlalchemy.engine.base.Engine: SQLAlchemy engine object
    """
    try:
        if db_url is None:
            # Use a default connection string if no URL is provided
            db_url = 'postgresql+psycopg2://postgres:83326874@localhost:5432/Stocks'

        # Create and return the engine
        engine = create_engine(db_url, pool_pre_ping=True)
        return engine

    except Exception as e:
        logging.error(f"Error creating database engine: {e}")
        raise


def fetch_stock_data(symbol, db_engine=None):
    """
    Fetch stock data for a given symbol from the database.

    Args:
        symbol (str): Stock symbol
        db_engine (sqlalchemy.engine.base.Engine, optional): SQLAlchemy database engine. 
                                                            If None, a new engine will be created.

    Returns:
        pd.DataFrame: Stock data with date as the index and 'Close' column
    """
    try:
        # Use provided engine or create a new one
        if db_engine is None:
            db_engine = get_db_engine()

        # Query the database
        query = f"""
        SELECT date, close 
        FROM stock_data 
        WHERE stock_symbol = '{symbol}' 
        ORDER BY date
        """
        stock_data = pd.read_sql(query, db_engine, index_col='date', parse_dates=['date'])

        # Validate the fetched data
        if stock_data.empty:
            logging.warning(f"No data found for symbol: {symbol}")
            return None

        # Rename columns to match expected format
        stock_data.rename(columns={'close': 'Close'}, inplace=True)

        logging.info(f"Fetched {len(stock_data)} rows of data for symbol: {symbol}")
        return stock_data

    except SQLAlchemyError as e:
        logging.error(f"Database error while fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while fetching data for {symbol}: {e}")
        return None