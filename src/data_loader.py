"""
Data loading module for AlphaInsights
"""
import pandas as pd
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_news_data(self):
        """Load news data from CSV file"""
        file_path = os.path.join(self.config.DATA_DIR, self.config.NEWS_DATA_FILE)
        try:
            df = pd.read_csv(file_path)
            
            # Convert date column and make timezone-naive
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Map company names to stock tickers
            if 'stock' in df.columns:
                df['stock_ticker'] = df['stock']
                ticker_to_company = {v: k for k, v in self.config.COMPANY_TICKER_MAP.items()}
                df['company_name'] = df['stock'].map(ticker_to_company).fillna(df['stock'])
            else:
                df['stock_ticker'] = df['company_name'].apply(self.extract_ticker_from_company)
                df = df.dropna(subset=['stock_ticker'])

            print(f"✅ Successfully loaded {len(df)} news articles")
            print(f"📊 Unique tickers found: {df['stock_ticker'].nunique()}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ News data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            return pd.DataFrame()

    def extract_ticker_from_company(self, company_name):
        """Extract ticker symbol from company name"""
        for company, ticker in self.config.COMPANY_TICKER_MAP.items():
            if company.lower() in company_name.lower():
                return ticker
        return None

    def load_stock_data(self, symbol, period="1y"):
        """Load stock price data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return pd.DataFrame()

            hist.reset_index(inplace=True)
            
            # Remove timezone info if present
            if 'Date' in hist.columns and hist['Date'].dt.tz is not None:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            hist['symbol'] = symbol
            return hist
            
        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_stock_data(self):
        """Load stock data for all tracked symbols"""
        news_data = self.load_news_data()
        if news_data.empty:
            print("❌ No news data available to extract tickers")
            return pd.DataFrame()

        unique_tickers = news_data['stock_ticker'].unique()
        print(f"📈 Loading stock data for {len(unique_tickers)} tickers...")

        all_data = []
        for symbol in unique_tickers:
            print(f"  Loading {symbol}...")
            data = self.load_stock_data(symbol)
            if not data.empty:
                all_data.append(data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Successfully loaded stock data for {combined_data['symbol'].nunique()} symbols")
            return combined_data
        else:
            print("❌ No stock data loaded")
            return pd.DataFrame()
