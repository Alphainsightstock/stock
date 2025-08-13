"""
Hybrid ML model combining LSTM and Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HybridMLModel:
    def __init__(self, config):
        self.config = config
        self.lstm_model = None
        self.rf_model = None
        self.is_trained = False
        
    def prepare_lstm_data(self, df, lookback_days=60):
        """Prepare data for LSTM model"""
        X, y = [], []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('Date')
            if len(symbol_data) < lookback_days + 1:
                continue
                
            prices = symbol_data['Close'].values
            
            for i in range(lookback_days, len(prices)):
                X.append(prices[i-lookback_days:i])
                # Target: 1 if price goes up tomorrow, 0 if down
                if i < len(prices) - 1:
                    next_price = prices[i + 1]
                    current_price = prices[i]
                    y.append(1 if next_price > current_price else 0)
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(self.config.LSTM_UNITS, return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.config.LSTM_DROPOUT),
            LSTM(self.config.LSTM_UNITS, return_sequences=True),
            Dropout(self.config.LSTM_DROPOUT),
            LSTM(self.config.LSTM_UNITS),
            Dropout(self.config.LSTM_DROPOUT),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def prepare_rf_data(self, df):
        """Prepare data for Random Forest model"""
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        df_sorted = df.sort_values(['symbol', 'Date'])
        
        # Create target variable (next day price direction)
        df_sorted['target'] = df_sorted.groupby('symbol')['Close'].shift(-1) > df_sorted['Close']
        df_sorted['target'] = df_sorted['target'].astype(int)
        
        # Remove last row for each symbol (no target available)
        df_clean = df_sorted.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        df_clean = df_clean.dropna(subset=available_features + ['target'])
        
        if df_clean.empty:
            return pd.DataFrame(), pd.Series()
        
        X = df_clean[available_features]
        y = df_clean['target']
        
        return X, y
    
    def train_models(self, df):
        """Train both LSTM and Random Forest models"""
        print("Preparing data for training...")
        
        # Prepare LSTM data
        X_lstm, y_lstm = self.prepare_lstm_data(df, self.config.LSTM_LOOKBACK_DAYS)
        
        # Prepare Random Forest data
        X_rf, y_rf = self.prepare_rf_data(df)
        
        if len(X_lstm) == 0 and len(X_rf) == 0:
            print("Insufficient data for training!")
            return
        
        # Train LSTM if data available
        if len(X_lstm) > 0:
            try:
                print(f"Training LSTM model with {len(X_lstm)} samples...")
                X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
                    X_lstm, y_lstm, test_size=0.2, random_state=42)
                
                self.lstm_model = self.build_lstm_model((X_lstm_train.shape[1], 1))
                X_lstm_train_reshaped = X_lstm_train.reshape((X_lstm_train.shape[0], 
                                                             X_lstm_train.shape[1], 1))
                X_lstm_test_reshaped = X_lstm_test.reshape((X_lstm_test.shape[0], 
                                                           X_lstm_test.shape[1], 1))
                
                self.lstm_model.fit(X_lstm_train_reshaped, y_lstm_train, 
                                   epochs=50, batch_size=32, verbose=0,
                                   validation_data=(X_lstm_test_reshaped, y_lstm_test))
                
                # Evaluate LSTM
                lstm_pred = (self.lstm_model.predict(X_lstm_test_reshaped, verbose=0) > 0.5).astype(int)
                lstm_accuracy = accuracy_score(y_lstm_test, lstm_pred)
                print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training LSTM: {e}")
                self.lstm_model = None
        
        # Train Random Forest if data available
        if len(X_rf) > 0:
            try:
                print(f"Training Random Forest model with {len(X_rf)} samples...")
                X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
                    X_rf, y_rf, test_size=0.2, random_state=42)
                
                self.rf_model = RandomForestClassifier(
                    n_estimators=self.config.RF_N_ESTIMATORS,
                    max_depth=self.config.RF_MAX_DEPTH,
                    random_state=42
                )
                self.rf_model.fit(X_rf_train, y_rf_train)
                
                # Evaluate Random Forest
                rf_pred = self.rf_model.predict(X_rf_test)
                rf_accuracy = accuracy_score(y_rf_test, rf_pred)
                print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training Random Forest: {e}")
                self.rf_model = None
        
        self.is_trained = (self.lstm_model is not None) or (self.rf_model is not None)
        
    def predict_hybrid_score(self, stock_data, symbol):
        """Generate hybrid prediction score"""
        if not self.is_trained:
            return 0.5
        
        try:
            # Get latest data for the symbol
            symbol_data = stock_data[stock_data['symbol'] == symbol].sort_values('Date')
            
            if symbol_data.empty:
                return 0.5
            
            lstm_prob = 0.5
            rf_prob = 0.5
            
            # LSTM prediction
            if self.lstm_model and len(symbol_data) >= self.config.LSTM_LOOKBACK_DAYS:
                lstm_input = symbol_data['Close'].tail(self.config.LSTM_LOOKBACK_DAYS).values
                lstm_input = lstm_input.reshape((1, len(lstm_input), 1))
                lstm_prob = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            
            # Random Forest prediction
            if self.rf_model:
                feature_columns = [
                    'MA_10', 'MA_30', 'RSI', 'BB_width', 'daily_return', 
                    'volatility', 'final_sentiment', 'article_count'
                ]
                available_features = [col for col in feature_columns if col in symbol_data.columns]
                
                if available_features and not symbol_data.empty:
                    latest_features = symbol_data[available_features].iloc[-1].values.reshape(1, -1)
                    rf_prob = self.rf_model.predict_proba(latest_features)[0][1]
            
            # Combine predictions (weighted average)
            if self.lstm_model and self.rf_model:
                hybrid_score = 0.6 * rf_prob + 0.4 * lstm_prob
            elif self.rf_model:
                hybrid_score = rf_prob
            elif self.lstm_model:
                hybrid_score = lstm_prob
            else:
                hybrid_score = 0.5
                
            return hybrid_score
            
        except Exception as e:
            print(f"Error in prediction for {symbol}: {e}")
            return 0.5
    
    def save_models(self, model_dir):
        """Save trained models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if self.lstm_model:
            self.lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))
        
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(model_dir, 'rf_model.pkl'))
    
    def load_models(self, model_dir):
        """Load trained models"""
        try:
            lstm_path = os.path.join(model_dir, 'lstm_model.h5')
            rf_path = os.path.join(model_dir, 'rf_model.pkl')
            
            if os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                
            self.is_trained = (self.lstm_model is not None) or (self.rf_model is not None)
            
            if self.is_trained:
                print("Models loaded successfully!")
            else:
                print("No trained models found!")
                
        except Exception as e:
            print(f"Error loading models: {e}")
