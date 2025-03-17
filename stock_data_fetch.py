import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import time
import logging


class StockDataFetcher:
    """A class to fetch stock data from Yahoo Finance and store it in PostgreSQL."""

    def __init__(self):
        """Initialize the StockDataFetcher with default parameters."""
        self.logger = logging.getLogger(__name__)
        self.db_engine = self.create_database_engine()

    def create_database_engine(self):
        """Create and return a database engine for PostgreSQL."""
        try:
            engine = create_engine('postgresql+psycopg2://postgres:83326874@localhost:5432/Stocks')
            self.logger.info("Database connection established.")
            return engine
        except Exception as e:
            self.logger.error(f"Failed to connect to the database: {e}")
            raise

    def fetch_yfinance_data(self, symbol: str, start_date='2023-01-01', end_date=pd.Timestamp.today().strftime('%Y-%m-%d'), max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance with comprehensive error handling.

        Args:
            symbol (str): Stock symbol to fetch.
            start_date (str, optional): Start date for data fetch, default is '2023-01-01'.
            end_date (str, optional): End date for data fetch, default is today's date.
            max_retries (int): Number of times to retry the download.

        Returns:
            pd.DataFrame: DataFrame containing the stock data.
        """
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                try:
                    # First, try Ticker.history()
                    stock_data = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        timeout=30,
                        raise_errors=True
                    )
                except Exception as e:
                    # If Ticker.history() fails, try yf.download()
                    self.logger.warning(f"Ticker.history() failed for {symbol}: {e}")
                    stock_data = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date,
                        timeout=30
                    )

                if stock_data.empty:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    continue

                stock_data.reset_index(inplace=True)
                stock_data['Symbol'] = symbol
                stock_data.columns = [col.replace(' ', '_').lower() for col in stock_data.columns]

                return stock_data

            except Exception as e:
                self.logger.warning(f"Error fetching data for {symbol} (Attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        self.logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
        return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the stock data."""
        try:
            if df.empty:
                self.logger.warning("DataFrame is empty")
                return False

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
                return False

            if df.isnull().sum().any():
                self.logger.warning("DataFrame contains null values")
                return False

            if (df['close'] <= 0).any():
                self.logger.warning("DataFrame contains non-positive close prices")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            return False

    def store_stock_data(self, df: pd.DataFrame) -> None:
        """Store stock data in PostgreSQL database."""
        try:
            columns_to_keep = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns_to_keep if col in df.columns]]
            df = df.rename(columns={'symbol': 'stock_symbol'})
            df.to_sql('stock_data', self.db_engine, if_exists='append', index=False)
            self.logger.info(f"Stored data for {df['stock_symbol'].iloc[0]}")

        except Exception as e:
            self.logger.error(f"Failed to store data in PostgreSQL: {e}")
            raise


def get_sp500_symbols() -> list:
    """Fetch S&P 500 stock symbols from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        return table['Symbol'].tolist()
    except Exception as e:
        logging.error(f"Error fetching S&P 500 symbols: {e}")
        return []


def fetch_and_store_sp500_stocks(chunk_size=50):
    """Fetch and store historical data for all S&P 500 stocks."""
    data_fetcher = StockDataFetcher()
    symbols = get_sp500_symbols()
    failed_symbols = []

    for i in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[i:i+chunk_size]
        for symbol in chunk_symbols:
            try:
                stock_data = data_fetcher.fetch_yfinance_data(symbol)
                if stock_data.empty or not data_fetcher.validate_data(stock_data):
                    failed_symbols.append(symbol)
                    continue
                data_fetcher.store_stock_data(stock_data)
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                failed_symbols.append(symbol)

        time.sleep(5)

    if failed_symbols:
        logging.error(f"Failed symbols: {failed_symbols}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('stock_data_fetch.log')
        ]
    )

    try:
        fetch_and_store_sp500_stocks()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")
