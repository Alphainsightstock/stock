"""
Streamlit dashboard for the sentiment trading project
"""
import streamlit as st
import sys
import os

# Fix import path for src modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from src folder
from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_data_processor import StockDataProcessor
from src.hybrid_model import HybridMLModel
from src.signal_generator import SignalGenerator
import config

# Page configuration
st.set_page_config(
    page_title="Sentiment Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sentiment-Based Stock Trading Dashboard")
st.markdown("*Analyzing Indian stocks using news sentiment and hybrid ML models*")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Load data button
if st.sidebar.button("🚀 Load Data & Train Model", type="primary"):
    with st.spinner("Loading data and training models... This may take a few minutes."):
        try:
            # Initialize components
            data_loader = DataLoader(config)
            sentiment_analyzer = SentimentAnalyzer()
            stock_processor = StockDataProcessor()
            hybrid_model = HybridMLModel(config)
            signal_generator = SignalGenerator(config)
            
            # Load and process data
            news_data = data_loader.load_news_data()
            print(f"News data loaded: {len(news_data)} articles")
            
            if news_data.empty:
                st.error("No news data found! Please check the CSV file.")
                st.stop()
                
            news_with_sentiment = sentiment_analyzer.analyze_news_sentiment(news_data)
            aggregated_sentiment = sentiment_analyzer.aggregate_stock_sentiment(news_with_sentiment)
            
            stock_data = data_loader.get_all_stock_data()
            print(f"Stock price data loaded: {len(stock_data)} records")
            
            if stock_data.empty:
                st.error("No stock data found!")
                st.stop()
                
            stock_data_processed = stock_processor.calculate_technical_indicators(stock_data)
            merged_data = stock_processor.merge_with_sentiment(stock_data_processed, aggregated_sentiment)
            final_data = stock_processor.prepare_features(merged_data)
            
            print(f"Final processed data contains: {len(final_data)} records")
            print(f"Available stocks: {final_data['symbol'].unique() if not final_data.empty else 'No stocks'}")
            
            final_data_scaled = stock_processor.scale_features(final_data)
            
            # Train model
            hybrid_model.train_models(final_data_scaled)
            
            # Generate signals
            signals = signal_generator.generate_signals(final_data_scaled, hybrid_model)
            print(f"Signals generated for stocks: {signals['symbol'].unique() if not signals.empty else 'No signals'}")
            
            # Store in session state
            st.session_state.final_data = final_data
            st.session_state.signals = signals
            st.session_state.news_data = news_with_sentiment
            st.session_state.sentiment_summary = sentiment_analyzer.get_latest_sentiment_summary(aggregated_sentiment)
            st.session_state.data_loaded = True
            
            st.success("✅ Data loaded and model trained successfully!")
            st.experimental_rerun()  # Refresh to show stock dropdown
            
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            print(f"Full error: {e}")

# Display available stocks in sidebar
if st.session_state.data_loaded and 'final_data' in st.session_state and not st.session_state.final_data.empty:
    available_stocks = st.session_state.final_data['symbol'].unique()
    selected_stock = st.sidebar.selectbox("📊 Select Stock for Analysis", available_stocks)
else:
    selected_stock = st.sidebar.selectbox("📊 Select Stock for Analysis", ["Load data first..."], disabled=True)

# Main dashboard
if st.session_state.data_loaded:
    
    # Trading Signals Summary
    st.subheader("🎯 Trading Signals Summary")
    
    if 'signals' in st.session_state and not st.session_state.signals.empty:
        col1, col2, col3 = st.columns(3)
        
        buy_count = len(st.session_state.signals[st.session_state.signals['signal'] == 'BUY'])
        sell_count = len(st.session_state.signals[st.session_state.signals['signal'] == 'SELL'])
        hold_count = len(st.session_state.signals[st.session_state.signals['signal'] == 'HOLD'])
        
        with col1:
            st.metric("🟢 BUY Signals", buy_count)
        with col2:
            st.metric("🔴 SELL Signals", sell_count)
        with col3:
            st.metric("🟡 HOLD Signals", hold_count)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Stock Analysis - {selected_stock}")
        
        if selected_stock != "Load data first..." and 'final_data' in st.session_state:
            # Get stock data for selected symbol
            stock_data = st.session_state.final_data[
                st.session_state.final_data['symbol'] == selected_stock
            ].sort_values('Date')
            
            if not stock_data.empty:
                # Create price and sentiment chart
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Stock Price with Moving Averages', 'Technical Indicators', 'Sentiment Analysis'),
                    vertical_spacing=0.08,
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Price chart with moving averages
                fig.add_trace(
                    go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                if 'MA_10' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data['MA_10'],
                            mode='lines',
                            name='MA 10',
                            line=dict(color='orange', dash='dash')
                        ),
                        row=1, col=1
                    )
                
                if 'MA_30' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data['MA_30'],
                            mode='lines',
                            name='MA 30',
                            line=dict(color='red', dash='dot')
                        ),
                        row=1, col=1
                    )
                
                # RSI
                if 'RSI' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple')
                        ),
                        row=2, col=1
                    )
                    
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Sentiment chart
                if 'final_sentiment' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data['final_sentiment'],
                            mode='lines+markers',
                            name='Sentiment Score',
                            line=dict(color='green'),
                            fill='tonexty'
                        ),
                        row=3, col=1
                    )
                
                fig.update_layout(height=800, showlegend=True, title_text=f"Complete Analysis - {selected_stock}")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning(f"No data available for {selected_stock}")
    
    with col2:
        st.subheader("🎯 Current Signals")
        
        if 'signals' in st.session_state and not st.session_state.signals.empty:
            # Display signals with enhanced styling
            for _, signal in st.session_state.signals.iterrows():
                if signal['signal'] == 'BUY':
                    color = "#00ff00"
                    icon = "🟢"
                elif signal['signal'] == 'SELL':
                    color = "#ff0000"
                    icon = "🔴"
                else:
                    color = "#ffaa00"
                    icon = "🟡"
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba(255,255,255,0.1);">
                        <h4>{icon} {signal['symbol']}</h4>
                        <p><strong>Signal:</strong> <span style="color: {color}; font-weight: bold;">{signal['signal']}</span></p>
                        <p><strong>Confidence:</strong> {signal['confidence']:.3f}</p>
                        <p><strong>Sentiment:</strong> {signal['sentiment_score']:.3f}</p>
                        <p><strong>Reason:</strong> {signal['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No signals generated yet. Please load data first.")
    
    # Sentiment Analysis Section
    st.subheader("📰 Sentiment Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Overall Sentiment Summary**")
        if 'sentiment_summary' in st.session_state and not st.session_state.sentiment_summary.empty:
            sentiment_df = st.session_state.sentiment_summary
            st.dataframe(sentiment_df, use_container_width=True)
        else:
            st.info("No sentiment summary available")
    
    with col2:
        st.write("**Recent News Analysis**")
        if selected_stock != "Load data first..." and 'news_data' in st.session_state:
            # Map ticker back to company name for news filtering
            ticker_to_company = {v: k for k, v in config.COMPANY_TICKER_MAP.items()}
            company_names = [k for k, v in config.COMPANY_TICKER_MAP.items() if v == selected_stock]
            
            if company_names and not st.session_state.news_data.empty:
                recent_news = st.session_state.news_data[
                    st.session_state.news_data['company_name'].isin(company_names)
                ].sort_values('date', ascending=False).head(5)
                
                if not recent_news.empty:
                    for _, news in recent_news.iterrows():
                        sentiment_color = "#00ff00" if news['final_sentiment'] > 0 else "#ff0000" if news['final_sentiment'] < 0 else "#888888"
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid {sentiment_color}; padding: 10px; margin: 5px 0; background-color: rgba(255,255,255,0.05); border-radius: 5px;">
                            <p><strong>{news['headline']}</strong></p>
                            <p style="font-size: 0.8em;">Date: {news['date'].strftime('%Y-%m-%d')}</p>
                            <p style="font-size: 0.8em;">Sentiment: <span style="color: {sentiment_color};">{news['final_sentiment']:.3f}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news found for this stock")
            else:
                st.info("No recent news found for this stock")

else:
    # Welcome screen when data is not loaded
    st.info("👈 Please click 'Load Data & Train Model' in the sidebar to start the analysis.")
    
    # Show sample data preview
    st.subheader("📋 Sample Data Preview")
    
    try:
        data_loader = DataLoader(config)
        news_data = data_loader.load_news_data()
        
        if not news_data.empty:
            st.write("**Sample News Data with Extracted Tickers:**")
            preview_columns = ['date', 'company_name', 'stock_ticker', 'headline', 'sentiment_label']
            available_columns = [col for col in preview_columns if col in news_data.columns]
            preview_data = news_data[available_columns].head(10)
            st.dataframe(preview_data, use_container_width=True)
        else:
            st.error("Sample data file not found! Please ensure 'data/sample_news_data.csv' exists.")
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
    
    # System overview
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    This advanced sentiment-based trading system includes:
    
    **📊 Data Processing Pipeline:**
    - 📰 News headline sentiment analysis using VADER + Financial keywords
    - 🏢 Company name to stock ticker mapping
    - 📈 Technical indicators (RSI, Moving Averages, Bollinger Bands)
    
    **🤖 Hybrid ML Model:**
    - 🔗 LSTM for time series pattern recognition
    - 🌳 Random Forest for feature-based predictions
    - ⚖️ Ensemble approach for robust signals
    
    **🎯 Trading Signal Generation:**
    - 🟢 **BUY**: High confidence + positive sentiment
    - 🔴 **SELL**: High confidence + negative sentiment  
    - 🟡 **HOLD**: Low confidence or neutral sentiment
    
    **📈 Supported Indian Stocks:**
    """)
    
    # Display supported stocks in a nice format
    if hasattr(config, 'STOCK_SYMBOLS'):
        stocks_per_row = 4
        stock_symbols = sorted(config.STOCK_SYMBOLS)
        
        for i in range(0, len(stock_symbols), stocks_per_row):
            cols = st.columns(stocks_per_row)
            for j, stock in enumerate(stock_symbols[i:i+stocks_per_row]):
                if j < len(cols):
                    with cols[j]:
                        st.code(stock)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    💡 <strong>Sentiment Trading System</strong> | Built with Streamlit, TensorFlow, and scikit-learn<br>
    ⚠️ <em>This is for educational purposes only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)
