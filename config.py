"""
Configuration file for the sentiment trading project
"""

# Company name to ticker mapping for Indian stocks
COMPANY_TICKER_MAP = {
    'Tata Consultancy Services': 'TCS.NS',
    'TCS': 'TCS.NS',
    'Infosys Limited': 'INFY.NS',
    'Infosys': 'INFY.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'RIL': 'RELIANCE.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'SBI': 'SBIN.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Larsen & Toubro': 'LT.NS',
    'L&T': 'LT.NS',
    'Wipro Limited': 'WIPRO.NS',
    'Wipro': 'WIPRO.NS',
    'HCL Technologies': 'HCLTECH.NS',
    'HCL Tech': 'HCLTECH.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'HUL': 'HINDUNILVR.NS',
    'ITC Limited': 'ITC.NS',
    'ITC': 'ITC.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Tata Motors': 'TATAMOTORS.NS',
    'Hero MotoCorp': 'HEROMOTOCO.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Sun Pharma': 'SUNPHARMA.NS',
    'Dr Reddys Laboratories': 'DRREDDY.NS',
    "Dr Reddy's": 'DRREDDY.NS',
    'Cipla Limited': 'CIPLA.NS',
    'Cipla': 'CIPLA.NS',
    'Lupin Limited': 'LUPIN.NS',
    'Lupin': 'LUPIN.NS',
    'Titan Company': 'TITAN.NS',
    'Titan': 'TITAN.NS',
    'UltraTech Cement': 'ULTRACEMCO.NS',
    'Grasim Industries': 'GRASIM.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'Tata Steel': 'TATASTEEL.NS',
    'Hindalco Industries': 'HINDALCO.NS',
    'Coal India': 'COALINDIA.NS',
    'NTPC Limited': 'NTPC.NS',
    'Power Grid Corp': 'POWERGRID.NS',
    'PowerGrid': 'POWERGRID.NS',
    'Oil and Natural Gas': 'ONGC.NS',
    'ONGC': 'ONGC.NS',
    'Indian Oil Corporation': 'IOC.NS',
    'IOC': 'IOC.NS',
    'Bharat Petroleum': 'BPCL.NS',
    'BPCL': 'BPCL.NS',
    'Hindustan Petroleum': 'HINDPETRO.NS',
    'HPCL': 'HINDPETRO.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Godrej Consumer Products': 'GODREJCP.NS',
    'Britannia Industries': 'BRITANNIA.NS',
    'Nestle India': 'NESTLEIND.NS',
    'Dabur India': 'DABUR.NS',
    'Marico Limited': 'MARICO.NS',
    'Colgate Palmolive': 'COLPAL.NS',
    'Bajaj Finserv': 'BAJAJFINSV.NS',
    'SBI Life Insurance': 'SBILIFE.NS',
    'HDFC Life Insurance': 'HDFCLIFE.NS',
    'ICICI Prudential': 'ICICIPRULI.NS',
    'ICICI Pru': 'ICICIPRULI.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Vedanta Limited': 'VEDL.NS',
    'JSW Energy': 'JSWENERGY.NS',
    'Adani Ports': 'ADANIPORTS.NS',
    'Jindal Steel': 'JINDALSTEL.NS',
    'Bajaj Holdings': 'BAJAJHLDNG.NS',
    'Mahindra Finance': 'MAHINDRA.NS',
    'Shree Cement': 'SHREECEM.NS',
    'ACC Limited': 'ACC.NS',
    'Ambuja Cements': 'AMBUJACEM.NS',
    'UPL Limited': 'UPL.NS',
    'Pidilite Industries': 'PIDILITIND.NS',
    'Berger Paints': 'BERGEPAINT.NS',
    'Eicher Motors': 'EICHERMOT.NS',
    'TVS Motor': 'TVSMOTOR.NS',
    'Mahindra Logistics': 'MAHLOG.NS',
    'Blue Dart Express': 'BLUEDART.NS',
    'InterGlobe Aviation': 'INDIGO.NS',
    'SpiceJet Limited': 'SPICEJET.NS',
    'Indian Railway Catering': 'IRCTC.NS',
    'IRCTC': 'IRCTC.NS',
    'Container Corporation': 'CONCOR.NS',
    'CONCOR': 'CONCOR.NS',
    'Petronet LNG': 'PETRONET.NS',
    'GAIL India': 'GAIL.NS',
    'Torrent Pharmaceuticals': 'TORNTPHARM.NS',
    'Aurobindo Pharma': 'AUROPHARMA.NS',
    'Divis Laboratories': 'DIVISLAB.NS',
    'Biocon Limited': 'BIOCON.NS',
    'Mindtree Limited': 'MINDTREE.NS',
    'Mphasis Limited': 'MPHASIS.NS',
    'L&T Infotech': 'LTI.NS',
    'Persistent Systems': 'PERSISTENT.NS',
    'HDFC Asset Management': 'HDFCAMC.NS',
    'SBI Cards': 'SBICARD.NS',
    'Bandhan Bank': 'BANDHANBNK.NS',
    'IndusInd Bank': 'INDUSINDBK.NS',
    'Yes Bank': 'YESBANK.NS',
    'Federal Bank': 'FEDERALBNK.NS',
    'ICICI Securities': 'ISEC.NS',
    'Motilal Oswal': 'MOTILALOFS.NS',
    'Zee Entertainment': 'ZEEL.NS',
    'Sun TV Network': 'SUNTV.NS'
}

# All tracked stock symbols
STOCK_SYMBOLS = list(set(COMPANY_TICKER_MAP.values()))

# Sentiment thresholds
POSITIVE_SENTIMENT_THRESHOLD = 0.1
NEGATIVE_SENTIMENT_THRESHOLD = -0.1

# Trading signal thresholds
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

# Model parameters
LSTM_LOOKBACK_DAYS = 60
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# File paths
DATA_DIR = "data"
MODEL_DIR = "models"
NEWS_DATA_FILE = "sample_news_data.csv"

# Additional configuration for enhanced functionality
TECHNICAL_INDICATORS = {
    'MA_SHORT': 10,
    'MA_LONG': 30,
    'RSI_PERIOD': 14,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    'VOLATILITY_PERIOD': 20
}

# Risk management parameters
RISK_MANAGEMENT = {
    'MAX_POSITION_SIZE': 0.1,  # 10% of portfolio
    'STOP_LOSS_PCT': 0.05,     # 5% stop loss
    'TAKE_PROFIT_PCT': 0.15,   # 15% take profit
    'MAX_DRAWDOWN': 0.2        # 20% max drawdown
}

# Feature importance weights for signal generation
FEATURE_WEIGHTS = {
    'SENTIMENT': 0.3,
    'TECHNICAL': 0.4,
    'ML_PREDICTION': 0.3
}

# Logging configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_FILE': 'trading_system.log'
}
