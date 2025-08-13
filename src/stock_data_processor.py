"""
Stock data processing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
import warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for stock data"""
        if df.empty:
            return df
        
        print("📈 Calculating technical indicators...")
        
        df = df.copy()
        df.sort_values(['symbol', 'Date'], inplace=True)
        
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) < 50:  # Need minimum data for indicators
                continue
                
            try:
                # Moving averages
                symbol_df['MA_10'] = ta.trend.sma_indicator(symbol_df['Close'], window=10)
                symbol_df['MA_30'] = ta.trend.sma_indicator(symbol_df['Close'], window=30)
                
                # RSI
                symbol_df['RSI'] = ta.momentum.rsi(symbol_df['Close'], window=14)
                
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(symbol_df['Close'])
                symbol_df['BB_upper'] = bollinger.bollinger_hband()
                symbol_df['BB_lower'] = bollinger.bollinger_lband()
                symbol_df['BB_middle'] = bollinger.bollinger_mavg()
                symbol_df['BB_width'] = symbol_df['BB_upper'] - symbol_df['BB_lower']
                
                # MACD
                macd = ta.trend.MACD(symbol_df['Close'])
                symbol_df['MACD'] = macd.macd()
                symbol_df['MACD_signal'] = macd.macd_signal()
                
                # Daily returns
                symbol_df['daily_return'] = symbol_df['Close'].pct_change()
                
                # Volatility
                symbol_df['volatility'] = symbol_df['daily_return'].rolling(window=20).std()
                
                # Volume indicators
                symbol_df['volume_sma'] = ta.volume.volume_sma(
                    symbol_df['Close'], symbol_df['Volume'], window=20
                )
                
                results.append(symbol_df)
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")
                continue
        
        if results:
            final_df = pd.concat(results, ignore_index=True)
            print(f"✅ Technical indicators calculated for {len(results)} symbols")
            return final_df
        else:
            print("❌ No technical indicators calculated")
            return df
    
    def merge_with_sentiment(self, stock_df, sentiment_df):
        """Merge stock data with sentiment data"""
        if stock_df.empty or sentiment_df.empty:
            print("⚠️ Empty dataframes provided for merging")
            return stock_df
        
        print("🔗 Merging stock data with sentiment data...")
        
        # Ensure both dataframes have timezone-naive datetime
        stock_df = stock_df.copy()
        sentiment_df = sentiment_df.copy()
        
        # Handle timezone conversion for stock data
        if 'Date' in stock_df.columns:
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            if stock_df['Date'].dt.tz is not None:
                stock_df['Date'] = stock_df['Date'].dt.tz_localize(None)
        
        # Handle timezone conversion for sentiment data
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            if sentiment_df['date'].dt.tz is not None:
                sentiment_df['date'] = sentiment_df['date'].dt.tz_localize(None)
        
        # Rename columns for consistency
        sentiment_df_renamed = sentiment_df.rename(columns={
            'stock_ticker': 'symbol',
            'date': 'Date'
        })
        
        # Merge on symbol and date
        merged = pd.merge(
            stock_df, 
            sentiment_df_renamed, 
            on=['symbol', 'Date'], 
            how='left'
        )
        
        # Fill missing sentiment values
        sentiment_cols = ['avg_sentiment', 'weighted_sentiment', 'article_count', 'avg_confidence']
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Create final sentiment score
        if 'avg_sentiment' in merged.columns:
            merged['final_sentiment'] = merged['avg_sentiment']
        
        print(f"✅ Merged data shape: {merged.shape}")
        
        return merged
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        if df.empty:
            return df
        
        print("🔧 Preparing features for ML models...")
        
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'MACD', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count', 'avg_confidence'
        ]
        
        # Select only available columns
        available_columns = ['symbol', 'Date', 'Close'] + [col for col in feature_columns if col in df.columns]
        
        # Ensure essential sentiment columns exist
        if 'final_sentiment' not in df.columns and 'avg_sentiment' in df.columns:
            df['final_sentiment'] = df['avg_sentiment']
            available_columns.append('final_sentiment')
        
        if 'article_count' not in df.columns:
            df['article_count'] = 0
            available_columns.append('article_count')
        
        if 'avg_confidence' not in df.columns:
            df['avg_confidence'] = 0.5
            available_columns.append('avg_confidence')
        
        df_features = df[available_columns].copy()
        
        # Remove rows with insufficient technical indicator data
        technical_cols = [col for col in ['MA_10', 'MA_30', 'RSI', 'BB_width'] if col in df_features.columns]
        if technical_cols:
            df_features = df_features.dropna(subset=technical_cols)
        
        print(f"✅ Prepared features: {df_features.shape}")
        
        return df_features
    
    def scale_features(self, df, fit=True):
        """Scale numerical features for ML models"""
        if df.empty:
            return df
        
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'MACD', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count', 'avg_confidence'
        ]
        
        available_feature_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_feature_columns:
            return df
        
        df_scaled = df.copy()
        
        if fit:
            df_scaled[available_feature_columns] = self.scaler.fit_transform(
                df[available_feature_columns].fillna(0)
            )
        else:
            df_scaled[available_feature_columns] = self.scaler.transform(
                df[available_feature_columns].fillna(0)
            )
        
        return df_scaled
