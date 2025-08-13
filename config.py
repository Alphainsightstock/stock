"""
Configuration file for AlphaInsights Sentiment Trading System
"""
import os

# Data Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
NEWS_DATA_FILE = "sample_news_data.csv"

# Indian Stock Symbols (NSE)
COMPANY_TICKER_MAP = {
    'Tata Consultancy Services': 'TCS.NS',
    'Infosys': 'INFY.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'Wipro': 'WIPRO.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Nestle India': 'NESTLEIND.NS',
    'Dr Reddys Labs': 'DRREDDY.NS'
}

# Sentiment Analysis Configuration
POSITIVE_SENTIMENT_THRESHOLD = 0.1
NEGATIVE_SENTIMENT_THRESHOLD = -0.1

# Trading Signal Configuration
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

# ML Model Configuration
LSTM_LOOKBACK_DAYS = 60
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# All tracked stock symbols
STOCK_SYMBOLS = list(set(COMPANY_TICKER_MAP.values()))
