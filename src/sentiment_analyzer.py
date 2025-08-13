"""
Enhanced sentiment analysis module with text processing
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        self.positive_keywords = [
            'growth', 'profit', 'revenue', 'expansion', 'partnership', 'approval', 
            'launch', 'increase', 'strong', 'optimistic', 'recovery', 'win', 
            'success', 'innovative', 'bullish', 'upgrade', 'breakthrough'
        ]
        
        self.negative_keywords = [
            'decline', 'loss', 'challenge', 'pressure', 'warning', 'issue',
            'problem', 'struggle', 'concern', 'risk', 'fall', 'drop',
            'bearish', 'downgrade', 'uncertainty', 'disruption'
        ]
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s\-\.\%\$]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def calculate_keyword_sentiment(self, text):
        cleaned_text = self.clean_text(text)
        positive_count = sum(1 for word in self.positive_keywords if word in cleaned_text)
        negative_count = sum(1 for word in self.negative_keywords if word in cleaned_text)
        if positive_count == 0 and negative_count == 0:
            return 0.0
        total_words = len(cleaned_text.split())
        positive_ratio = positive_count / max(total_words, 1)
        negative_ratio = negative_count / max(total_words, 1)
        return (positive_ratio - negative_ratio) * 2
    
    def get_sentiment_score(self, text):
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return 0.0
        vader_scores = self.analyzer.polarity_scores(cleaned_text)
        vader_compound = vader_scores['compound']
        keyword_sentiment = self.calculate_keyword_sentiment(text)
        final_score = 0.7 * vader_compound + 0.3 * keyword_sentiment
        return max(-1.0, min(1.0, final_score))
    
    def analyze_news_sentiment(self, news_df):
        news_df = news_df.copy()
        news_df['headline_sentiment'] = news_df['headline'].apply(self.get_sentiment_score)
        label_to_score = {'positive': 0.6, 'neutral': 0.0, 'negative': -0.6}
        news_df['manual_sentiment'] = news_df['sentiment_label'].map(label_to_score).fillna(0.0)
        news_df['final_sentiment'] = (0.8 * news_df['headline_sentiment'] + 0.2 * news_df['manual_sentiment'])
        news_df['sentiment_confidence'] = 1 - abs(news_df['headline_sentiment'] - news_df['manual_sentiment'])
        return news_df
    
    def aggregate_stock_sentiment(self, news_df, window_days=1):
        news_df['date'] = pd.to_datetime(news_df['date'])
        grouped = news_df.groupby(['stock_ticker', 'date']).agg({
            'final_sentiment': ['mean', 'std', 'count'],
            'sentiment_confidence': 'mean',
            'headline': lambda x: ' | '.join(x)
        }).reset_index()
        grouped.columns = [
            'stock_ticker', 'date', 'avg_sentiment', 'sentiment_std', 
            'article_count', 'avg_confidence', 'combined_headlines'
        ]
        grouped['sentiment_std'] = grouped['sentiment_std'].fillna(0)
        grouped['weighted_sentiment'] = grouped['avg_sentiment'] * grouped['avg_confidence'] * np.log1p(grouped['article_count'])
        return grouped
    
    def get_latest_sentiment_summary(self, aggregated_df):
        latest_sentiment = aggregated_df.sort_values('date').groupby('stock_ticker').tail(1)
        summary = []
        for _, row in latest_sentiment.iterrows():
            sentiment_label = 'Positive' if row['avg_sentiment'] > 0.1 else 'Negative' if row['avg_sentiment'] < -0.1 else 'Neutral'
            summary.append({
                'ticker': row['stock_ticker'],
                'sentiment_score': round(row['avg_sentiment'], 3),
                'sentiment_label': sentiment_label,
                'confidence': round(row['avg_confidence'], 3),
                'article_count': int(row['article_count']),
                'latest_date': row['date'].strftime('%Y-%m-%d')
            })
        return pd.DataFrame(summary)
