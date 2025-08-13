import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta

class StockDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def calculate_technical_indicators(self, df):
        df = df.copy()
        results = []
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            if len(sdf) < 30: continue
            sdf['MA_10'] = ta.trend.sma_indicator(sdf['Close'], window=10)
            sdf['MA_30'] = ta.trend.sma_indicator(sdf['Close'], window=30)
            sdf['RSI'] = ta.momentum.rsi(sdf['Close'], window=14)
            results.append(sdf)
        return pd.concat(results, ignore_index=True) if results else df

    def merge_with_sentiment(self, stock_df, sent_df):
        stock_df = stock_df.copy()
        sent_df = sent_df.copy()
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
        sent_df['date'] = pd.to_datetime(sent_df['date']).dt.tz_localize(None)
        sent_df = sent_df.rename(columns={'stock_ticker':'symbol','date':'Date'})
        merged = pd.merge(stock_df, sent_df, on=['symbol','Date'], how='left')
        merged['final_sentiment'] = merged.get('avg_sentiment', 0).fillna(0)
        return merged

    def prepare_features(self, df):
        features = ['MA_10','MA_30','RSI','final_sentiment']
        cols = ['symbol','Date','Close'] + features
        return df[cols].dropna()

    def scale_features(self, df):
        features = ['MA_10','MA_30','RSI','final_sentiment']
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.fit_transform(df[features])
        return df_scaled
