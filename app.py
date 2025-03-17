from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Set paths for forecasts, analysis, and visualizations
FORECAST_PATH = 'C:/Users/niraj/CS539 ML PROJECT/analysis/forecasts'
ANALYSIS_PATH = 'C:/Users/niraj/CS539 ML PROJECT/analysis/performance_metrics'
VISUALIZATION_PATH = 'C:/Users/niraj/CS539 ML PROJECT/analysis/visualizations'

# Simulating a list of available stock symbols
stocks = [
    'AXON', 'BKR', 'EBAY', 'FICO', 'AOS', 'ARE', 'APTV', 'AVY', 'AZO', 'BAC',
    'BALL', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BKNG', 'BKR', 'BLDR', 'EPAM',
    'ELV', 'BLK', 'BR', 'CAG', 'CB', 'CAT', 'CCL', 'D', 'DG', 'DIS', 'DOV', 'EOG',
    'FITB', 'FFIBV', 'FOXA', 'FRT', 'GD', 'GE', 'GEN', 'GEV', 'GLD', 'HAS', 'HCA', 'HD',
    'HLT', 'INCY', 'IP', 'IPG', 'J', 'JNJ', 'JPM', 'JHY', 'KHC', 'KIM', 'KLAC', 'KMB',
    'CDOS', 'LIN', 'LQQ', 'LOW', 'NEE', 'NVR', 'OMC', 'OTIS', 'OXY', 'PCAR', 'PAYC',
    'PAYX', 'PEG', 'REG', 'REGN', 'RMD', 'ROP', 'XOM', 'WYNN', 'ZTS'
]

# Route to serve static files
@app.route('/static/<folder>/<filename>')
def serve_static(folder, filename):
    folder_path = {
        'forecasts': FORECAST_PATH,
        'performance_metrics': ANALYSIS_PATH,
        'visualizations': VISUALIZATION_PATH
    }
    return send_from_directory(folder_path[folder], filename)

# Home route
@app.route('/')
def index():
    return render_template('index.html', stocks=stocks)

# Results route
@app.route('/results', methods=['POST'])
def results():
    data = request.get_json()
    stock_symbol = data['stock_symbol']

    # File names for forecasts and analysis
    arima_forecast_file = f"{stock_symbol}_arima_forecast.csv"
    sarima_forecast_file = f"{stock_symbol}_sarima_forecast.csv"
    prophet_forecast_file = f"{stock_symbol}_prophet_forecast.csv"
    analysis_file = f"{stock_symbol}_metrics.csv"

    # URLs for visualizations
    visualizations = {
        'Price Analysis': {'url': f"/static/visualizations/{stock_symbol}_price_analysis.png", 'width': 1000, 'height': 700},
        'Trend Analysis': {'url': f"/static/visualizations/{stock_symbol}_trend_analysis.png", 'width': 1000, 'height': 700},
        'Seasonality Analysis': {'url': f"/static/visualizations/{stock_symbol}_seasonality_analysis.png", 'width': 1000, 'height': 700}
    }

    # Check if forecast and analysis files exist
    missing_files = []
    forecast_files = [arima_forecast_file, sarima_forecast_file, prophet_forecast_file]
    for forecast_file in forecast_files:
        if not os.path.exists(os.path.join(FORECAST_PATH, forecast_file)):
            missing_files.append(forecast_file)

    if not os.path.exists(os.path.join(ANALYSIS_PATH, analysis_file)):
        missing_files.append(analysis_file)

    if missing_files:
        return jsonify({"error": f"Missing files: {', '.join(missing_files)}"}), 404

    # Return data for the frontend
    return jsonify({
        'arima_forecast': f"/static/forecasts/{arima_forecast_file}",
        'sarima_forecast': f"/static/forecasts/{sarima_forecast_file}",
        'prophet_forecast': f"/static/forecasts/{prophet_forecast_file}",
        'analysis': f"/static/performance_metrics/{analysis_file}",
        'visualizations': visualizations
    })


if __name__ == '__main__':
    app.run(debug=True)
