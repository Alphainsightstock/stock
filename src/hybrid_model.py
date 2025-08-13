"""
Hybrid ML model combining Random Forest and basic prediction logic
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HybridMLModel:
    def __init__(self, config):
        self.config = config
        self.rf_model = None
        self.is_trained = False
        
    def prepare_training_data(self, df):
        """Prepare data for training"""
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        print("🎯 Preparing training data...")
        
        feature_columns = [
            'MA_10', 'MA_30', 'RSI', 'BB_width', 'MACD', 'daily_return', 
            'volatility', 'final_sentiment', 'article_count', 'avg_confidence'
        ]
        
        # Only use columns that exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            print("❌ No features available for training")
            return pd.DataFrame(), pd.Series()
        
        df_sorted = df.sort_values(['symbol', 'Date'])
        
        # Create target variable (next day price direction)
        df_sorted['next_close'] = df_sorted.groupby('symbol')['Close'].shift(-1)
        df_sorted['target'] = (df_sorted['next_close'] > df_sorted['Close']).astype(int)
        
        # Remove last row for each symbol (no target available)
        df_clean = df_sorted.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        df_clean = df_clean.dropna(subset=available_features + ['target'])
        
        if df_clean.empty:
            print("❌ No clean data available for training")
            return pd.DataFrame(), pd.Series()
        
        X = df_clean[available_features]
        y = df_clean['target']
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train_model(self, df):
        """Train the Random Forest model"""
        print("🤖 Starting model training...")
        
        X, y = self.prepare_training_data(df)
        
        if X.empty or len(X) < 10:
            print("❌ Insufficient data for training")
            return
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest
            print("🌲 Training Random Forest model...")
            self.rf_model = RandomForestClassifier(
                n_estimators=self.config.RF_N_ESTIMATORS,
                max_depth=self.config.RF_MAX_DEPTH,
                random_state=42,
                class_weight='balanced'
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Accuracy: {accuracy:.4f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("🔍 Top 5 Most Important Features:")
            print(feature_importance.head().to_string(index=False))
            
            self.is_trained = True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            self.rf_model = None
            self.is_trained = False
    
    def predict_direction(self, stock_data, symbol):
        """Predict price direction for a stock"""
        if not self.is_trained or self.rf_model is None:
            return 0.5  # Neutral prediction if model not trained
        
        try:
            # Get latest data for the symbol
            symbol_data = stock_data[stock_data['symbol'] == symbol].sort_values('Date')
            
            if symbol_data.empty:
                return 0.5
            
            # Prepare features
            feature_columns = [
                'MA_10', 'MA_30', 'RSI', 'BB_width', 'MACD', 'daily_return', 
                'volatility', 'final_sentiment', 'article_count', 'avg_confidence'
            ]
            
            available_features = [col for col in feature_columns if col in symbol_data.columns]
            
            if not available_features:
                return 0.5
            
            # Use latest row
            latest_features = symbol_data[available_features].iloc[-1:].fillna(0)
            
            # Predict probability
            probability = self.rf_model.predict_proba(latest_features)[0]
            
            # Return probability of price increase
            return probability if len(probability) > 1 else 0.5
            
        except Exception as e:
            print(f"❌ Error predicting for {symbol}: {e}")
            return 0.5
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if not self.is_trained or self.rf_model is None:
            print("❌ No trained model to save")
            return
        
        if filepath is None:
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            filepath = os.path.join(self.config.MODEL_DIR, 'hybrid_model.pkl')
        
        try:
            joblib.dump({
                'model': self.rf_model,
                'is_trained': self.is_trained
            }, filepath)
            print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'hybrid_model.pkl')
        
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return
        
        try:
            model_data = joblib.load(filepath)
            self.rf_model = model_data['model']
            self.is_trained = model_data['is_trained']
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
