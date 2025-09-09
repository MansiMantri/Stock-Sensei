# Current paths (incorrect)
FORECAST_PATH = '/analysis/forecasts'
ANALYSIS_PATH = '/analysis/performance_metrics'
VISUALIZATION_PATH = '/analysis/visualizations'

# Should be changed to (correct)
FORECAST_PATH = os.path.join(os.path.dirname(__file__), 'analysis', 'forecasts')
ANALYSIS_PATH = os.path.join(os.path.dirname(__file__), 'analysis', 'performance_metrics')
VISUALIZATION_PATH = os.path.join(os.path.dirname(__file__), 'analysis', 'visualizations')