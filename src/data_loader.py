import pandas as pd
import yfinance as yf
import os

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_news_data(self):
        file_path = os.path.join(self.config.DATA_DIR, self.config.NEWS_DATA_FILE)
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            # Map company name → ticker
            if 'stock' in df.columns:
                df['stock_ticker'] = df['stock']
                ticker_to_company = {v:k for k,v in self.config.COMPANY_TICKER_MAP.items()}
                df['company_name'] = df['stock'].map(ticker_to_company).fillna(df['stock'])
            else:
                df['stock_ticker'] = df['company_name'].apply(self.extract_ticker_from_company)
                df = df.dropna(subset=['stock_ticker'])

            return df

        except FileNotFoundError:
            print(f"CSV not found at {file_path}")
            return pd.DataFrame()

    def extract_ticker_from_company(self, name):
        if pd.isna(name):
            return None
        for company, ticker in self.config.COMPANY_TICKER_MAP.items():
            if company.lower() in name.lower():
                return ticker
        return None

    def load_stock_data(self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            if hist.empty: return pd.DataFrame()
            hist.reset_index(inplace=True)
            if 'Date' in hist.columns and hist['Date'].dt.tz is not None:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            hist['symbol'] = symbol
            return hist
        except Exception as e:
            print("Stock data error:", e)
            return pd.DataFrame()

    def get_all_stock_data(self):
        news_df = self.load_news_data()
        if news_df.empty:
            return pd.DataFrame()
        tickers = news_df['stock_ticker'].unique()
        all_data = []
        for t in tickers:
            df = self.load_stock_data(t)
            if not df.empty:
                all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
