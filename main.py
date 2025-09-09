import os
import sys
import logging
import traceback
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Set matplotlib backend before other imports
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Import model functions
from models.arima_model import arima_model
from models.sarima_model import sarima_model
from models.prophet_model import prophet_model
from data_analysis import StockDataAnalyzer
from utils.comparison_utils import compare_models

def setup_logging(debug=False):
    """
    Configure comprehensive logging with file and console handlers.
    
    Args:
        debug (bool): Enable debug logging
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'stock_analysis_{timestamp}.log'

    # Logging configuration
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Create logger
logger = logging.getLogger(__name__)

class StockAnalysisPipeline:
    def __init__(self, db_url=None, debug=False):
        """
        Initialize Stock Analysis Pipeline.
        
        Args:
            db_url (str, optional): Database connection URL
            debug (bool, optional): Enable debug mode
        """
        try:
            # Set debug mode
            self.debug = debug
            setup_logging(debug)

            # Determine database URL
            if db_url is None:
                db_url = os.getenv(
                    'DATABASE_URL', 
                    'postgresql+psycopg2://postgres:Mansi%407038@localhost:5432/stock'
                )

            # Create database engine
            self.db_engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                echo=debug
            )

            # Validate database connection
            self.validate_database()

            # Setup analysis directories
            self.base_dir = Path('analysis')
            self.setup_directories()

            # Define forecasting models
            self.models = {
                'ARIMA': arima_model,
                'SARIMA': sarima_model,
                'Prophet': prophet_model
            }

            logger.info("Stock Analysis Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def validate_database(self):
        """Validate database connection and schema."""
        try:
            with self.db_engine.connect() as conn:
                # Test basic connection
                conn.execute(text("SELECT 1"))

                # Check stock_data table exists
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_name = 'stock_data')"
                ))
                if not result.scalar():
                    raise ValueError("stock_data table does not exist")

                # Check forecast tables exist
                forecast_tables = [
                    'stock_forecasts_arima', 
                    'stock_forecasts_sarima', 
                    'stock_forecasts_prophet'
                ]
                
                for table in forecast_tables:
                    result = conn.execute(text(
                        f"SELECT EXISTS (SELECT FROM information_schema.tables "
                        f"WHERE table_name = '{table}')"
                    ))
                    if not result.scalar():
                        logger.warning(f"Forecast table {table} does not exist")

        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            raise

    def setup_directories(self):
        """Create necessary directories for analysis outputs."""
        directories = [
            self.base_dir / 'forecasts',
            self.base_dir / 'performance_metrics',
            self.base_dir / 'visualizations',
            self.base_dir / 'reports',
            self.base_dir / 'logs'
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_stock_data(self, symbol):
        """
        Fetch and prepare stock data for analysis.
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            pd.DataFrame: Prepared stock data
        """
        try:
            # Explicit column selection in query
            query = text("""
                SELECT 
                    date,
                    close as "Close"
                FROM stock_data 
                WHERE stock_symbol = :symbol 
                ORDER BY date
            """)
            
            # Fetch data
            df = pd.read_sql(
                query,
                self.db_engine,
                params={'symbol': symbol},
                parse_dates=['date']
            )

            # Validate data
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Set index and clean data
            df = df.set_index('date')
            df = df.sort_index()
            df = df.dropna()

            # Check for sufficient data
            if len(df) < 30:
                logger.warning(f"Insufficient data points for {symbol}: {len(df)}")
                return None

            # Debug logging
            if self.debug:
                logger.debug(f"Processed data for {symbol}:")
                logger.debug(f"Columns: {df.columns.tolist()}")
                logger.debug(f"Shape: {df.shape}")
                logger.debug(f"First few rows:\n{df.head()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def analyze_stock(self, symbol):
        """
        Perform comprehensive analysis for a single stock.
        
        Args:
            symbol (str): Stock symbol
        """
        try:
            logger.info(f"Starting analysis for {symbol}")

            # Get and validate stock data
            stock_data = self.get_stock_data(symbol)
            if stock_data is None:
                logger.warning(f"Skipping analysis for {symbol}: No valid data")
                return

            # Generate analysis report with visualizations
            try:
                analyzer = StockDataAnalyzer(stock_data, symbol)
                analysis_report = analyzer.generate_analysis_report()
                
                # Log visualization paths
                if self.debug:
                    logger.debug(f"Visualizations for {symbol}:")
                    for viz_type, path in analysis_report.get('visualization_files', {}).items():
                        logger.debug(f"{viz_type}: {path}")
            
            except Exception as analysis_error:
                logger.error(f"Error generating analysis report for {symbol}: {analysis_error}")
                logger.error(traceback.format_exc())
                return

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            logger.error(traceback.format_exc())

    def run_pipeline(self, symbols_limit=None):
        """
        Execute the complete analysis pipeline.
        
        Args:
            symbols_limit (int, optional): Limit number of stocks to process
        """
        try:
            # Get list of symbols
            query = text("SELECT DISTINCT stock_symbol FROM stock_data")
            if symbols_limit:
                query = text(f"{query} LIMIT :limit")
                
            with self.db_engine.connect() as conn:
                if symbols_limit:
                    result = conn.execute(query, {'limit': symbols_limit})
                else:
                    result = conn.execute(query)
                symbols = [row[0] for row in result]

            logger.info(f"Processing {len(symbols)} stocks")

            # Process stocks with thread pool
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
                futures = {executor.submit(self.analyze_stock, symbol): symbol 
                         for symbol in symbols}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to process {symbol}: {e}")

            logger.info("Analysis pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(traceback.format_exc())

def main():
    """Main entry point for the Stock Analysis Pipeline."""
    parser = argparse.ArgumentParser(description="Stock Analysis Pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--limit", type=int, help="Limit number of stocks to process")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = StockAnalysisPipeline(debug=args.debug)
        
        # Run pipeline or test mode
        if args.test:
            # Test analysis for specific symbol
            test_symbol = args.symbol or "AAPL"
            logger.info(f"Running analysis test for symbol: {test_symbol}")
            pipeline.analyze_stock(test_symbol)
        else:
            # Normal pipeline execution
            pipeline.run_pipeline(symbols_limit=args.limit)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
