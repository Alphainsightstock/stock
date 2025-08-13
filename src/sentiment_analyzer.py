"""
News sentiment analysis using VADER sentiment analyzer
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Custom finance-specific lexicon additions
        finance_lexicon = {
            'profit': 2.0, 'loss': -2.0, 'gain': 1.5, 'drop': -1.5,
            'bull': 2.0, 'bear': -2.0, 'rally': 1.8, 'crash': -2.5,
            'upgrade': 1.8, 'downgrade': -1.8, 'buy': 1.5, 'sell': -1.5,
            'growth': 1.5, 'decline': -1.5, 'surge': 2.0, 'plunge': -2.0,
            'earnings': 1.0, 'dividend': 1.2, 'bankruptcy': -2.8,
            'merger': 1.0, 'acquisition': 1.0, 'layoffs': -1.8
        }
        
        # Update VADER lexicon with finance terms
        self.analyzer.lexicon.update(finance_lexicon)

    def get_sentiment_score(self, text):
        """Get sentiment score for a single text"""
        if pd.isna(text) or text.strip() == '':
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        # Use compound score which ranges from -1 to 1
        return scores['compound']

    def get_detailed_sentiment(self, text):
        """Get detailed sentiment breakdown"""
        if pd.isna(text) or text.strip() == '':
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        return self.analyzer.polarity_scores(text)

    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment for all news headlines"""
        if news_df.empty:
            return news_df
        
        news_df = news_df.copy()
        
        print("🧠 Analyzing news sentiment...")
        
        # Analyze headline sentiment
        news_df['headline_sentiment'] = news_df['headline'].apply(self.get_sentiment_score)
        
        # Convert manual labels to numerical scores if available
        if 'sentiment_label' in news_df.columns:
            label_mapping = {'positive': 0.6, 'negative': -0.6, 'neutral': 0.0}
            news_df['manual_sentiment'] = news_df['sentiment_label'].map(label_mapping).fillna(0.0)
            
            # Combine automated and manual sentiment (weighted average)
            news_df['final_sentiment'] = (0.7 * news_df['headline_sentiment'] + 
                                         0.3 * news_df['manual_sentiment'])
        else:
            news_df['final_sentiment'] = news_df['headline_sentiment']
        
        # Add sentiment confidence score
        news_df['sentiment_confidence'] = news_df['headline_sentiment'].abs()
        
        # Classify sentiment
        conditions = [
            news_df['final_sentiment'] > 0.1,
            news_df['final_sentiment'] < -0.1
        ]
        choices = ['Positive', 'Negative']
        news_df['sentiment_category'] = np.select(conditions, choices, default='Neutral')
        
        print(f"✅ Sentiment analysis complete!")
        print(f"📊 Sentiment distribution:")
        print(news_df['sentiment_category'].value_counts())
        
        return news_df

    def aggregate_stock_sentiment(self, news_with_sentiment):
        """Aggregate sentiment by stock and date"""
        if news_with_sentiment.empty:
            return pd.DataFrame()
        
        print("📊 Aggregating sentiment by stock and date...")
        
        # Group by stock ticker and date
        aggregated = news_with_sentiment.groupby(['stock_ticker', 'date']).agg({
            'final_sentiment': ['mean', 'std', 'count'],
            'sentiment_confidence': 'mean'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['stock_ticker', 'date', 'avg_sentiment', 'sentiment_std', 
                             'article_count', 'avg_confidence']
        
        # Fill NaN standard deviations with 0
        aggregated['sentiment_std'] = aggregated['sentiment_std'].fillna(0)
        
        # Calculate weighted sentiment (considering article count and confidence)
        aggregated['weighted_sentiment'] = (
            aggregated['avg_sentiment'] * 
            np.log1p(aggregated['article_count']) * 
            aggregated['avg_confidence']
        )
        
        print(f"✅ Aggregated sentiment for {len(aggregated)} stock-date combinations")
        
        return aggregated

    def get_latest_sentiment_summary(self, aggregated_sentiment):
        """Get latest sentiment summary for dashboard"""
        if aggregated_sentiment.empty:
            return {}
        
        latest_date = aggregated_sentiment['date'].max()
        latest_data = aggregated_sentiment[aggregated_sentiment['date'] == latest_date]
        
        return {
            'date': latest_date,
            'total_stocks': len(latest_data),
            'avg_sentiment': latest_data['avg_sentiment'].mean(),
            'positive_stocks': len(latest_data[latest_data['avg_sentiment'] > 0.1]),
            'negative_stocks': len(latest_data[latest_data['avg_sentiment'] < -0.1]),
            'neutral_stocks': len(latest_data[
                (latest_data['avg_sentiment'] >= -0.1) & 
                (latest_data['avg_sentiment'] <= 0.1)
            ])
        }
