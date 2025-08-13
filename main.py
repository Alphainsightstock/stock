import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_data_processor import StockDataProcessor
from src.hybrid_model import HybridMLModel
from src.signal_generator import SignalGenerator
import config

def main():
    print("🚀 Starting Sentiment-Based Trading System...")
    
    data_loader = DataLoader(config)
    sentiment_analyzer = SentimentAnalyzer()
    stock_processor = StockDataProcessor()
    hybrid_model = HybridMLModel(config)
    signal_generator = SignalGenerator(config)
    
    news_data = data_loader.load_news_data()
    if news_data.empty:
        print("❌ No news data found! Please check the CSV file.")
        return
    print(f"✅ Loaded {len(news_data)} news articles")
    
    news_with_sentiment = sentiment_analyzer.analyze_news_sentiment(news_data)
    aggregated_sentiment = sentiment_analyzer.aggregate_stock_sentiment(news_with_sentiment)
    print(f"✅ Analyzed sentiment for {len(aggregated_sentiment)} stock-date combinations")
    
    sentiment_summary = sentiment_analyzer.get_latest_sentiment_summary(aggregated_sentiment)
    print("\nLatest Sentiment Summary:")
    for _, row in sentiment_summary.iterrows():
        print(f"{row['ticker']:12} | {row['sentiment_label']:8} | Score: {row['sentiment_score']:6.3f}")
    
    stock_data = data_loader.get_all_stock_data()
    if stock_data.empty:
        print("❌ No stock data found!")
        return
    print(f"✅ Loaded data for {stock_data['symbol'].nunique()} stocks")
    
    stock_data_processed = stock_processor.calculate_technical_indicators(stock_data)
    merged_data = stock_processor.merge_with_sentiment(stock_data_processed, aggregated_sentiment)
    final_data = stock_processor.prepare_features(merged_data)
    
    if final_data.empty:
        print("❌ No data available after merging and cleaning!")
        return
    print(f"✅ Final dataset contains {len(final_data)} records")
    
    final_data_scaled = stock_processor.scale_features(final_data)
    hybrid_model.train_models(final_data_scaled)
    if not hybrid_model.is_trained:
        print("❌ Model training failed!")
        return
    print("✅ Hybrid model trained successfully")
    
    signals = signal_generator.generate_signals(final_data_scaled, hybrid_model)
    print("\n📊 TRADING SIGNALS GENERATED\n")
    
    for _, signal in signals.iterrows():
        print(f"{signal['symbol']:12} | {signal['signal']:4} | Confidence: {signal['confidence']:.3f} | Reason: {signal['reason']}")
    
    import os
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    hybrid_model.save_models(config.MODEL_DIR)
    print("\n✅ Models saved successfully")

if __name__ == "__main__":
    main()
