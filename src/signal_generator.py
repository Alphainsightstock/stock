"""
Trading signal generation based on sentiment and ML predictions
"""
import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_signals(self, stock_data, hybrid_model):
        """Generate trading signals for all stocks"""
        if stock_data.empty:
            return pd.DataFrame()
        
        print("🎯 Generating trading signals...")
        
        signals = []
        available_symbols = stock_data['symbol'].unique()
        
        for symbol in available_symbols:
            try:
                signal_data = self._generate_signal_for_stock(stock_data, hybrid_model, symbol)
                if signal_data:
                    signals.append(signal_data)
            except Exception as e:
                print(f"❌ Error generating signal for {symbol}: {e}")
                continue
        
        if signals:
            signals_df = pd.DataFrame(signals)
            print(f"✅ Generated {len(signals_df)} trading signals")
            
            # Print signal summary
            signal_counts = signals_df['signal'].value_counts()
            print("📊 Signal Distribution:")
            for signal, count in signal_counts.items():
                print(f"  {signal}: {count}")
            
            return signals_df
        else:
            print("❌ No signals generated")
            return pd.DataFrame()
    
    def _generate_signal_for_stock(self, stock_data, hybrid_model, symbol):
        """Generate signal for a single stock"""
        # Get stock data
        symbol_data = stock_data[stock_data['symbol'] == symbol].sort_values('Date')
        
        if symbol_data.empty:
            return None
        
        latest_data = symbol_data.iloc[-1]
        
        # Get sentiment score
        sentiment_score = latest_data.get('final_sentiment', 0)
        if pd.isna(sentiment_score):
            sentiment_score = 0
        
        # Get article count for confidence
        article_count = latest_data.get('article_count', 0)
        if pd.isna(article_count):
            article_count = 0
        
        # Get ML prediction
        ml_prediction = hybrid_model.predict_direction(stock_data, symbol)
        
        # Combine signals
        signal, confidence, reason = self._combine_signals(
            sentiment_score, ml_prediction, article_count, latest_data
        )
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'ml_prediction': ml_prediction,
            'article_count': article_count,
            'reason': reason,
            'timestamp': pd.Timestamp.now(),
            'current_price': latest_data.get('Close', 0)
        }
    
    def _combine_signals(self, sentiment_score, ml_prediction, article_count, latest_data):
        """Combine different signals into final recommendation"""
        
        # Base confidence on article count (more articles = more confidence)
        base_confidence = min(0.3 + (article_count * 0.1), 0.9)
        
        # Sentiment-based signal
        sentiment_signal = 'NEUTRAL'
        if sentiment_score > self.config.POSITIVE_SENTIMENT_THRESHOLD:
            sentiment_signal = 'POSITIVE'
        elif sentiment_score < self.config.NEGATIVE_SENTIMENT_THRESHOLD:
            sentiment_signal = 'NEGATIVE'
        
        # ML-based signal
        ml_signal = 'NEUTRAL'
        if ml_prediction > self.config.BUY_THRESHOLD:
            ml_signal = 'POSITIVE'
        elif ml_prediction < self.config.SELL_THRESHOLD:
            ml_signal = 'NEGATIVE'
        
        # Technical indicators (if available)
        technical_signal = self._get_technical_signal(latest_data)
        
        # Combine all signals
        positive_votes = 0
        negative_votes = 0
        
        if sentiment_signal == 'POSITIVE':
            positive_votes += 1
        elif sentiment_signal == 'NEGATIVE':
            negative_votes += 1
            
        if ml_signal == 'POSITIVE':
            positive_votes += 1
        elif ml_signal == 'NEGATIVE':
            negative_votes += 1
            
        if technical_signal == 'POSITIVE':
            positive_votes += 1
        elif technical_signal == 'NEGATIVE':
            negative_votes += 1
        
        # Final decision
        if positive_votes >= 2:
            final_signal = 'BUY'
            confidence = base_confidence * (positive_votes / 3)
            reason = f"Bullish consensus ({positive_votes}/3 indicators positive)"
        elif negative_votes >= 2:
            final_signal = 'SELL'
            confidence = base_confidence * (negative_votes / 3)
            reason = f"Bearish consensus ({negative_votes}/3 indicators negative)"
        else:
            final_signal = 'HOLD'
            confidence = 0.5
            reason = "Mixed signals - no clear consensus"
        
        return final_signal, confidence, reason
    
    def _get_technical_signal(self, data):
        """Generate signal based on technical indicators"""
        try:
            rsi = data.get('RSI', 50)
            ma_10 = data.get('MA_10', 0)
            ma_30 = data.get('MA_30', 0)
            current_price = data.get('Close', 0)
            
            positive_indicators = 0
            negative_indicators = 0
            
            # RSI signal
            if pd.notna(rsi):
                if rsi < 30:  # Oversold
                    positive_indicators += 1
                elif rsi > 70:  # Overbought
                    negative_indicators += 1
            
            # Moving average signal
            if pd.notna(ma_10) and pd.notna(ma_30) and ma_10 > 0 and ma_30 > 0:
                if ma_10 > ma_30:  # Short MA above long MA
                    positive_indicators += 1
                else:
                    negative_indicators += 1
            
            # Price vs MA signal
            if pd.notna(ma_10) and ma_10 > 0 and current_price > 0:
                if current_price > ma_10:
                    positive_indicators += 1
                else:
                    negative_indicators += 1
            
            if positive_indicators > negative_indicators:
                return 'POSITIVE'
            elif negative_indicators > positive_indicators:
                return 'NEGATIVE'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            return 'NEUTRAL'
