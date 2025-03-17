import os
import sys
import logging
import traceback

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical: Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class StockDataAnalyzer:
    def __init__(self, stock_data, symbol):
        """
        Initialize the StockDataAnalyzer with stock data and symbol.
        
        Parameters:
            stock_data (pd.DataFrame): DataFrame containing stock price data
            symbol (str): Stock symbol/ticker
        """
        # Validate input
        if not isinstance(stock_data, pd.DataFrame):
            raise ValueError("stock_data must be a pandas DataFrame")
        
        # Ensure 'Close' column exists
        if 'Close' not in stock_data.columns:
            raise ValueError("DataFrame must contain a 'Close' column")
        
        # Ensure index is datetime
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        
        self.stock_data = stock_data
        self.symbol = symbol
        
        # Ensure analysis directories exist
        self.analysis_dir = os.path.join('analysis', 'visualizations')
        os.makedirs(self.analysis_dir, exist_ok=True)

    def _safe_plot(self, plot_func, output_path):
        """
        Safe plot generation wrapper.
        
        Args:
            plot_func (callable): Function to generate plot
            output_path (str): Path to save plot
        
        Returns:
            str or None: Path to saved plot
        """
        try:
            # Clear any existing plots
            plt.close('all')
            
            # Create figure
            fig = plt.figure(figsize=(16, 12), dpi=300)
            
            # Generate plot
            plot_func(fig)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Verify file creation
            if os.path.exists(output_path):
                logger.info(f"Plot saved for {self.symbol}: {output_path}")
                logger.info(f"File size: {os.path.getsize(output_path)} bytes")
                return output_path
            else:
                logger.error(f"File was not created for {self.symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error in plot generation for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            plt.close('all')
            return None

    def _price_analysis_plot(self, fig):
        """
        Generate price analysis plot.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to plot on
        """
        # Ensure data is clean
        clean_data = self.stock_data.dropna()
        
        # Validate data
        if len(clean_data) < 2:
            logger.error(f"Insufficient data points for {self.symbol}")
            return
        
        # Calculate returns
        returns = clean_data['Close'].pct_change().dropna()
        
        # Create subplots
        axes = fig.subplots(2, 2)
        fig.suptitle(f'{self.symbol} Comprehensive Price Analysis', fontsize=16)
        
        # Price History
        axes[0, 0].plot(clean_data.index, clean_data['Close'], color='blue')
        axes[0, 0].set_title('Price History')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True)
        
        # Returns Distribution
        axes[0, 1].hist(returns, bins=50, edgecolor='black')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Frequency')
        
        # Rolling Statistics
        rolling_mean = clean_data['Close'].rolling(window=20).mean()
        rolling_std = clean_data['Close'].rolling(window=20).std()
        axes[1, 0].plot(clean_data.index, clean_data['Close'], label='Price', alpha=0.7)
        axes[1, 0].plot(clean_data.index, rolling_mean, label='20-day MA', color='red')
        axes[1, 0].fill_between(
            clean_data.index, 
            rolling_mean - rolling_std, 
            rolling_mean + rolling_std, 
            alpha=0.2, 
            color='red'
        )
        axes[1, 0].set_title('Price with Rolling Mean and Std')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Q-Q Plot
        stats.probplot(returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Returns')

    def plot_price_analysis(self):
        """
        Create comprehensive price analysis visualization.
        
        Returns:
            str: Path to saved visualization
        """
        try:
            logger.info(f"Starting price analysis plot for {self.symbol}")
            
            # Prepare output path
            output_path = os.path.join(
                self.analysis_dir, 
                f'{self.symbol}_price_analysis.png'
            )
            
            # Generate plot using safe method
            return self._safe_plot(self._price_analysis_plot, output_path)
        
        except Exception as e:
            logger.error(f"Comprehensive error in price analysis plot for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _trend_analysis_plot(self, fig):
        """
        Generate trend analysis plot.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to plot on
        """
        # Ensure data is clean
        clean_data = self.stock_data.dropna()
        
        # Validate data
        if len(clean_data) < 2:
            logger.error(f"Insufficient data points for {self.symbol}")
            return
        
        # Create subplot
        ax = fig.add_subplot(1, 1, 1)
        
        # Calculate moving averages
        ma50 = clean_data['Close'].rolling(window=50).mean()
        ma200 = clean_data['Close'].rolling(window=200).mean()
        
        # Plot price and moving averages
        ax.plot(
            clean_data.index, 
            clean_data['Close'], 
            label='Price', 
            color='blue',
            alpha=0.7
        )
        ax.plot(
            clean_data.index, 
            ma50, 
            label='50-day MA', 
            color='orange'
        )
        ax.plot(
            clean_data.index, 
            ma200, 
            label='200-day MA', 
            color='red'
        )
        
        ax.set_title(f'{self.symbol} Trend Analysis', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)

    def analyze_trends(self):
        """
        Analyze and visualize stock price trends.
        
        Returns:
            str: Path to saved trend analysis plot
        """
        try:
            logger.info(f"Starting trend analysis plot for {self.symbol}")
            
            # Prepare output path
            output_path = os.path.join(
                self.analysis_dir, 
                f'{self.symbol}_trend_analysis.png'
            )
            
            # Generate plot using safe method
            return self._safe_plot(self._trend_analysis_plot, output_path)
        
        except Exception as e:
            logger.error(f"Comprehensive error in trend analysis plot for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _seasonality_analysis_plot(self, fig):
        """
        Generate seasonality analysis plot.
        
        Args:
            fig (matplotlib.figure.Figure): Figure to plot on
        """
        # Ensure data is clean
        clean_data = self.stock_data.dropna()
        
        # Validate data
        if len(clean_data) < 2:
            logger.error(f"Insufficient data points for {self.symbol}")
            return
        
        # Create subplots
        axes = fig.subplots(1, 2)
        fig.suptitle(f'{self.symbol} Seasonality Analysis', fontsize=16)
        
        # Monthly Seasonality
        monthly_avg = clean_data.groupby(clean_data.index.month)['Close'].mean()
        monthly_avg.plot(kind='bar', color='green', ax=axes[0])
        axes[0].set_title('Monthly Price Seasonality')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Price')
        axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                                 rotation=45)
        
        # Yearly Trend
        yearly_avg = clean_data.groupby(clean_data.index.year)['Close'].mean()
        yearly_avg.plot(kind='line', marker='o', color='purple', ax=axes[1])
        axes[1].set_title('Yearly Price Trend')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Average Price')
        axes[1].grid(True)

    def analyze_seasonality(self):
        """
        Analyze and visualize seasonal patterns.
        
        Returns:
            str: Path to saved seasonality analysis plot
        """
        try:
            logger.info(f"Starting seasonality analysis plot for {self.symbol}")
            
            # Prepare output path
            output_path = os.path.join(
                self.analysis_dir, 
                f'{self.symbol}_seasonality_analysis.png'
            )
            
            # Generate plot using safe method
            return self._safe_plot(self._seasonality_analysis_plot, output_path)
        
        except Exception as e:
            logger.error(f"Comprehensive error in seasonality analysis plot for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def generate_analysis_report(self):
        """
        Generate a comprehensive analysis report.
        
        Returns:
            dict: Dictionary containing analysis results and visualization paths
        """
        try:
            logger.info(f"Starting analysis report generation for {self.symbol}")
            
            # Generate visualizations
            price_analysis_plot = self.plot_price_analysis()
            trend_analysis_plot = self.analyze_trends()
            seasonality_analysis_plot = self.analyze_seasonality()
            
            # Prepare basic statistics
            basic_stats = {
                'mean_price': self.stock_data['Close'].mean(),
                'median_price': self.stock_data['Close'].median(),
                'min_price': self.stock_data['Close'].min(),
                'max_price': self.stock_data['Close'].max(),
                'price_std': self.stock_data['Close'].std(),
                'total_observations': len(self.stock_data)
            }
            
            # Compile report
            report = {
                'symbol': self.symbol,
                'basic_statistics': basic_stats,
                'visualization_files': {
                    'price_analysis': price_analysis_plot,
                    'trend_analysis': trend_analysis_plot,
                    'seasonality_analysis': seasonality_analysis_plot
                }
            }
            
            # Verify visualization paths
            for viz_type, path in report['visualization_files'].items():
                if path is None or not os.path.exists(path):
                    logger.warning(f"{viz_type.capitalize()} visualization not generated for {self.symbol}")
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating analysis report for {self.symbol}: {e}")
            logger.error(traceback.format_exc())
            return {}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stock_analysis_debug.log', encoding='utf-8')
    ]
)