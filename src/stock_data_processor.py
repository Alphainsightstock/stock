"""
Stock data processing and feature engineering - Fixed timezone issue
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        df.sort_values(['symbol', 'Date'], inplace=True)
        
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Moving averages
            symbol_df['MA_10'] = symbol_df['Close'].rolling(10).mean()
            symbol_df['MA_30'] = symbol_df['Close'].rolling(30).mean()
            
            # RSI calculation
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            symbol_df['RSI'] = calculate_rsi(symbol_df['Close'])
            
            # Bollinger Bands
            symbol_df['BB_middle'] = symbol_df['MA_10']
            bb_std = symbol_df['Close'].rolling(10).std()
            symbol_df['BB_upper'] = symbol_df['BB_middle'] + (bb_std * 2)
            symbol_df['BB_lower'] = symbol_df['BB_middle'] - (bb_std * 2)
            symbol_df['BB_width'] = symbol_df['BB_upper'] - symbol_df['BB_lower']
            
            # Daily returns
            symbol_df['daily_return'] = symbol_df['Close'].pct_change()
            
            # Volatility (20-day rolling std of returns)
            symbol_df['volatility'] = symbol_df['daily_return'].rolling(20).std()
            
            results.append(symbol_df)
        
        return pd.concat(results, ignore_index=True)
    
    def merge_with_sentiment(self, stock_df, sentiment_df):
        """Merge stock data with sentiment data - Fixed timezone issue"""
        
        # Convert both datetime columns to timezone-naive
        stock_df = stock_df.copy()
        sentiment_df = sentiment_df.copy()
        
        # Handle timezone conversion for stock data
        if 'Date' in stock_df.columns:
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            # Remove timezone info if present
            if stock_df['Date'].dt.tz is not None:
                stock_df['Date'] = stock_df['Date'].dt.tz_localize(None)
        
        # Handle timezone conversion for sentiment data
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            # Remove timezone info if present
            if sentiment_df['date'].dt.tz is not None:
                sentiment_df['date'] = sentiment_df['date'].dt.tz_localize(None)
        
        # Rename columns for consistency
        sentiment_df_renamed = sentiment_df.rename(columns={
            'stock_ticker': 'symbol',
            'date': 'Date'
        })
        
        # Merge on symbol and date
        merged = pd.merge(stock_df, sentiment_df_renamed, 
                         on=['symbol', 'Date'], how='left')
        
        # Fill missing sentiment values with 0 (neutral)
        sentiment_cols = ['avg_sentiment', 'weighted_sentiment', 'article_count', 'avg_confidence']
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Rename sentiment columns for consistency
        if 'avg_sentiment' in merged.columns:
            merged['final_sentiment'] = merged['avg_sentiment']
        
        return merged
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count'
        ]
        
        # Select only rows with all required features
        required_columns = ['symbol', 'Date', 'Close'] + feature_columns
        available_columns = [col for col in required_columns if col in df.columns]
        
        if 'final_sentiment' not in df.columns and 'avg_sentiment' in df.columns:
            df['final_sentiment'] = df['avg_sentiment']
        
        if 'article_count' not in df.columns:
            df['article_count'] = 0
        
        df_features = df[available_columns].copy()
        df_features = df_features.dropna(subset=['MA_10', 'MA_30', 'RSI', 'BB_width'])
        
        return df_features
    
    def scale_features(self, df, fit=True):
        """Scale features for ML models"""
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count'
        ]
        
        # Only scale columns that exist
        available_feature_columns = [col for col in feature_columns if col in df.columns]
        
        df_scaled = df.copy()
        
        if available_feature_columns:
            if fit:
                df_scaled[available_feature_columns] = self.scaler.fit_transform(df[available_feature_columns].fillna(0))
            else:
                df_scaled[available_feature_columns] = self.scaler.transform(df[available_feature_columns].fillna(0))
        
        return df_scaled
