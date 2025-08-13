"""
Data loading and preprocessing module adapted for CSV with 'stock' column (ticker)
"""
import pandas as pd
import yfinance as yf
import os

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_news_data(self):
        """Load news data from CSV file and handle ticker/company name mapping"""
        file_path = os.path.join(self.config.DATA_DIR, self.config.NEWS_DATA_FILE)
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # If 'stock' column exists and 'company_name' doesn't, use ticker directly
            if 'stock' in df.columns and 'company_name' not in df.columns:
                df['stock_ticker'] = df['stock']  # Direct ticker from CSV
                
                # Reverse map ticker to company name for display (optional)
                ticker_to_company = {v: k for k, v in self.config.COMPANY_TICKER_MAP.items()}
                df['company_name'] = df['stock'].map(ticker_to_company).fillna(df['stock'])
            
            else:
                # Fallback: try to extract ticker from 'company_name'
                df['stock_ticker'] = df['company_name'].apply(self.extract_ticker_from_company)
                df = df.dropna(subset=['stock_ticker'])
            
            print(f"Successfully loaded {len(df)} news articles")
            print(f"Unique tickers found: {df['stock_ticker'].nunique()}")
            
            return df
            
        except FileNotFoundError:
            print(f"News data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()

    def extract_ticker_from_company(self, company_name):
        """Fallback: extract ticker from company name using config mapping (not used if 'stock' given)"""
        if pd.isna(company_name):
            return None
        return self.config.COMPANY_TICKER_MAP.get(company_name.strip(), None)

    def load_stock_data(self, symbol, period="1y"):
        """Load stock price data from yfinance by ticker symbol"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            if hist.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            hist.reset_index(inplace=True)
            hist['symbol'] = symbol
            return hist
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def get_all_stock_data(self):
        """Download stock data for all unique stock tickers in news data"""
        news_data = self.load_news_data()
        if news_data.empty:
            print("No news data available to extract tickers")
            return pd.DataFrame()

        unique_tickers = news_data['stock_ticker'].unique()
        print(f"Loading stock data for {len(unique_tickers)} tickers: {list(unique_tickers)}")

        all_data = []
        for symbol in unique_tickers:
            data = self.load_stock_data(symbol)
            if not data.empty:
                all_data.append(data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded stock data for {combined_data['symbol'].nunique()} symbols")
            return combined_data
        else:
            print("No stock data loaded.")
            return pd.DataFrame()
