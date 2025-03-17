CREATE TABLE stock_data (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10),
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
	adj_close FLOAT,
    volume BIGINT
);



SELECT * FROM stock_data LIMIT 10;
SELECT COUNT(*) FROM stock_data;

SELECT * FROM stock_forecasts_arima;
SELECT COUNT(*) FROM stock_forecasts_arima;

SELECT * FROM stock_forecasts_sarima;
SELECT COUNT(*) FROM stock_forecasts_sarima;

SELECT * FROM stock_forecasts_prophet;
SELECT COUNT(*) FROM stock_forecasts_prophet;


CREATE TABLE stock_forecasts_arima (
    date DATE,
    predicted_close FLOAT,
    stock_symbol VARCHAR(10)
);

CREATE TABLE stock_forecasts_sarima (
    date DATE,
    predicted_close FLOAT,
    stock_symbol VARCHAR(10)
);

CREATE TABLE stock_forecasts_prophet (
    date DATE,
    predicted_close FLOAT,
    stock_symbol VARCHAR(10)
);