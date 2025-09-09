# Stock Sensei 📈

Stock Sensei is an advanced stock analysis and forecasting platform that combines multiple prediction models to provide comprehensive market insights.

## Features 🚀

- Multiple forecasting models (ARIMA, SARIMA, Prophet)
- Interactive visualizations
- Performance metrics analysis
- Historical data analysis
- Real-time stock data fetching
- Web-based dashboard interface

## Tech Stack 💻

- Python
- Flask (Web Framework)
- SQLAlchemy (Database ORM)
- Pandas (Data Analysis)
- Plotly (Visualizations)
- Prophet, ARIMA, SARIMA (Forecasting Models)

## Project Structure 📁

```plaintext
├── app.py                 # Flask application entry point
├── main.py                # Core analysis pipeline
├── data_analysis.py       # Data analysis utilities
├── performance_metrics.py # Performance evaluation
├── stock_data_fetch.py    # Stock data retrieval
├── models/               # Forecasting models
│   ├── arima_model.py
│   ├── prophet_model.py
│   └── sarima_model.py
├── templates/            # HTML templates
│   ├── index.html
│   └── results.html
├── static/              # Static assets
│   └── style.css
└── utils/               # Utility functions
    ├── comparison_utils.py
    └── visualization_utils.py
```

## Setup Instructions 🛠️

1. Clone the repository:
```bash
git clone https://github.com/MansiMantri/Stock-Sensei.git
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.env\Scriptsctivate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL database and update connection string in `main.py`

5. Run the application:
```bash
python app.py
```

## Usage 📊

1. Access the web interface at `http://localhost:5000`
2. Enter stock symbol and select analysis parameters
3. View generated forecasts, visualizations, and performance metrics

## Features in Detail 🔍

### Multiple Forecasting Models
- ARIMA (Auto-Regressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Prophet (Facebook's time series forecasting tool)

### Comprehensive Analysis
- Historical price trends
- Volume analysis
- Technical indicators
- Performance metrics

### Interactive Visualizations
- Price forecasts
- Trend analysis
- Comparative model performance
- Error metrics

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [Prophet](https://facebook.github.io/prophet/)
- [Statsmodels](https://www.statsmodels.org/)
- [Plotly](https://plotly.com/)
- [Flask](https://flask.palletsprojects.com/)

---

Developed with ❤️ by Mansi Mantri
