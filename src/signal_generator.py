"""
Trading signal generation module
"""
import pandas as pd

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_signals(self, stock_data, hybrid_model):
        """Generate buy/sell/hold signals for all stocks"""
        signals = []
        
        # Get unique symbols from stock data
        available_symbols = stock_data['symbol'].unique()
        
        for symbol in available_symbols:
            # Check if we have data for this symbol
            symbol_data = stock_data[stock_data['symbol'] == symbol]
            if symbol_data.empty:
                continue
            
            # Get sentiment score
            latest_data = symbol_data.sort_values('Date').iloc[-1]
            sentiment_score = latest_data.get('final_sentiment', 0)
            if pd.isna(sentiment_score):
                sentiment_score = 0
            
            # Apply sentiment filter
            if sentiment_score <= self.config.POSITIVE_SENTIMENT_THRESHOLD:
                signal = 'HOLD'
                confidence = 0.3
                reason = f'Neutral/Negative sentiment ({sentiment_score:.3f})'
            else:
                # Get hybrid model prediction
                hybrid_score = hybrid_model.predict_hybrid_score(stock_data, symbol)
                
                # Generate signal based on hybrid score
                if hybrid_score >= self.config.BUY_THRESHOLD:
                    signal = 'BUY'
                    confidence = hybrid_score
                    reason = f'Strong bullish signal (Score: {hybrid_score:.3f})'
                elif hybrid_score <= self.config.SELL_THRESHOLD:
                    signal = 'SELL' 
                    confidence = 1 - hybrid_score
                    reason = f'Strong bearish signal (Score: {hybrid_score:.3f})'
                else:
                    signal = 'HOLD'
                    confidence = 0.5
                    reason = f'Neutral signal (Score: {hybrid_score:.3f})'
            
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'reason': reason,
                'timestamp': pd.Timestamp.now()
            })
        
        return pd.DataFrame(signals)
    
    def filter_positive_sentiment_stocks(self, sentiment_data):
        """Filter stocks with positive sentiment"""
        positive_stocks = sentiment_data[
            sentiment_data['avg_sentiment'] > self.config.POSITIVE_SENTIMENT_THRESHOLD
        ]['stock_ticker'].unique().tolist()
        
        return positive_stocks
