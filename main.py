"""
Main processing module for AlphaInsights Sentiment Trading System
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_data_processor import StockDataProcessor
from src.hybrid_model import HybridMLModel
from src.signal_generator import SignalGenerator
import config

def main():
    """Main processing function"""
    print("🚀 Starting AlphaInsights Sentiment Trading System...")
    print("=" * 60)
    
    # Initialize components
    data_loader = DataLoader(config)
    sentiment_analyzer = SentimentAnalyzer()
    stock_processor = StockDataProcessor()
    hybrid_model = HybridMLModel(config)
    signal_generator = SignalGenerator(config)
    
    try:
        # Step 1: Load news data
        print("\n1️⃣ Loading news data...")
        news_data = data_loader.load_news_data()
        
        if news_data.empty:
            print("❌ No news data found. Please check the CSV file.")
            return
        
        # Step 2: Analyze sentiment
        print("\n2️⃣ Analyzing sentiment...")
        news_with_sentiment = sentiment_analyzer.analyze_news_sentiment(news_data)
        aggregated_sentiment = sentiment_analyzer.aggregate_stock_sentiment(news_with_sentiment)
        
        # Step 3: Load stock data
        print("\n3️⃣ Loading stock price data...")
        stock_data = data_loader.get_all_stock_data()
        
        if stock_data.empty:
            print("❌ No stock data found. Check internet connection.")
            return
        
        # Step 4: Process technical indicators
        print("\n4️⃣ Calculating technical indicators...")
        stock_data_processed = stock_processor.calculate_technical_indicators(stock_data)
        
        # Step 5: Merge with sentiment data
        print("\n5️⃣ Merging stock and sentiment data...")
        merged_data = stock_processor.merge_with_sentiment(
            stock_data_processed, aggregated_sentiment
        )
        
        final_data = stock_processor.prepare_features(merged_data)
        
        if final_data.empty:
            print("❌ No data available after processing")
            return
        
        # Step 6: Train ML model
        print("\n6️⃣ Training ML model...")
        final_data_scaled = stock_processor.scale_features(final_data)
        hybrid_model.train_model(final_data_scaled)
        
        # Step 7: Generate signals
        print("\n7️⃣ Generating trading signals...")
        signals = signal_generator.generate_signals(final_data_scaled, hybrid_model)
        
        # Step 8: Display results
        print("\n8️⃣ Results Summary:")
        print("=" * 40)
        
        if not signals.empty:
            print(f"📊 Total signals generated: {len(signals)}")
            print("\n🎯 Trading Signals:")
            print("-" * 80)
            print(f"{'Symbol':<12} {'Signal':<6} {'Confidence':<12} {'Sentiment':<12} {'Reason'}")
            print("-" * 80)
            
            for _, row in signals.head(10).iterrows():
                print(f"{row['symbol']:<12} {row['signal']:<6} {row['confidence']:<12.2f} "
                      f"{row['sentiment_score']:<12.2f} {row['reason'][:30]}...")
            
            # Save results
            signals.to_csv('trading_signals.csv', index=False)
            print(f"\n💾 Signals saved to trading_signals.csv")
        else:
            print("❌ No trading signals generated")
        
        # Save model
        hybrid_model.save_model()
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error in main processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
