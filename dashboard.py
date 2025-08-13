"""
Streamlit dashboard for the sentiment trading project - Fixed dropdown issue
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
if 'loading_in_progress' not in st.session_state:
    st.session_state.loading_in_progress = False

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Load data button
if st.sidebar.button("🚀 Load Data & Train Model", type="primary", disabled=st.session_state.loading_in_progress):
    st.session_state.loading_in_progress = True
    
    # Clear previous data
    if 'final_data' in st.session_state:
        del st.session_state.final_data
    if 'signals' in st.session_state:
        del st.session_state.signals
    if 'news_data' in st.session_state:
        del st.session_state.news_data
    if 'sentiment_summary' in st.session_state:
        del st.session_state.sentiment_summary
    
    st.session_state.data_loaded = False
    
    with st.spinner("Loading data and training models... This may take a few minutes."):
        try:
            # Initialize components
            data_loader = DataLoader(config)
            sentiment_analyzer = SentimentAnalyzer()
            stock_processor = StockDataProcessor()
            hybrid_model = HybridMLModel(config)
            signal_generator = SignalGenerator(config)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load news data
            status_text.text('Loading news data...')
            progress_bar.progress(10)
            news_data = data_loader.load_news_data()
            
            if news_data.empty:
                st.error("❌ No news data found! Please check the CSV file at 'data/sample_news_data.csv'")
                st.session_state.loading_in_progress = False
                st.stop()
            
            # Step 2: Analyze sentiment
            status_text.text('Analyzing sentiment...')
            progress_bar.progress(25)
            news_with_sentiment = sentiment_analyzer.analyze_news_sentiment(news_data)
            aggregated_sentiment = sentiment_analyzer.aggregate_stock_sentiment(news_with_sentiment)
            
            # Step 3: Load stock data
            status_text.text('Loading stock price data...')
            progress_bar.progress(40)
            stock_data = data_loader.get_all_stock_data()
            
            if stock_data.empty:
                st.error("❌ No stock data found! Check internet connection or stock symbols.")
                st.session_state.loading_in_progress = False
                st.stop()
            
            # Step 4: Process technical indicators
            status_text.text('Calculating technical indicators...')
            progress_bar.progress(55)
            stock_data_processed = stock_processor.calculate_technical_indicators(stock_data)
            
            # Step 5: Merge data
            status_text.text('Merging data...')
            progress_bar.progress(70)
            merged_data = stock_processor.merge_with_sentiment(stock_data_processed, aggregated_sentiment)
            final_data = stock_processor.prepare_features(merged_data)
            
            if final_data.empty:
                st.error("❌ No data available after processing! Check your data files.")
                st.session_state.loading_in_progress = False
                st.stop()
            
            # Step 6: Scale features and train model
            status_text.text('Training ML models...')
            progress_bar.progress(85)
            final_data_scaled = stock_processor.scale_features(final_data)
            hybrid_model.train_models(final_data_scaled)
            
            # Step 7: Generate signals
            status_text.text('Generating trading signals...')
            progress_bar.progress(95)
            signals = signal_generator.generate_signals(final_data_scaled, hybrid_model)
            
            # Store in session state
            st.session_state.final_data = final_data
            st.session_state.signals = signals
            st.session_state.news_data = news_with_sentiment
            st.session_state.sentiment_summary = sentiment_analyzer.get_latest_sentiment_summary(aggregated_sentiment)
            st.session_state.data_loaded = True
            st.session_state.loading_in_progress = False
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text('Complete!')
            
            st.success("✅ Data loaded and model trained successfully!")
            
            # Force page refresh to update dropdown
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.session_state.loading_in_progress = False
            st.session_state.data_loaded = False

# Display available stocks in sidebar - FIXED VERSION
available_stocks = []
selected_stock = None

if st.session_state.data_loaded and 'final_data' in st.session_state:
    if not st.session_state.final_data.empty:
        available_stocks = sorted(st.session_state.final_data['symbol'].unique().tolist())
        selected_stock = st.sidebar.selectbox(
            "📊 Select Stock for Analysis", 
            available_stocks,
            key="stock_selector"
        )
    else:
        selected_stock = st.sidebar.selectbox(
            "📊 Select Stock for Analysis", 
            ["No stocks available"], 
            disabled=True
        )
else:
    selected_stock = st.sidebar.selectbox(
        "📊 Select Stock for Analysis", 
        ["Load data first..."], 
        disabled=True
    )

# Show loading status
if st.session_state.loading_in_progress:
    st.sidebar.info("🔄 Loading in progress...")

# Main dashboard content
if st.session_state.data_loaded and selected_stock and selected_stock not in ["Load data first...", "No stocks available"]:
    
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
        
        # Show signal for selected stock
        stock_signal = st.session_state.signals[st.session_state.signals['symbol'] == selected_stock]
        if not stock_signal.empty:
            signal_info = stock_signal.iloc[0]
            signal_color = "#00ff00" if signal_info['signal'] == 'BUY' else "#ff0000" if signal_info['signal'] == 'SELL' else "#ffaa00"
            
            st.markdown(f"""
            ### Current Signal for {selected_stock}
            <div style="border: 3px solid {signal_color}; border-radius: 10px; padding: 20px; margin: 10px 0; text-align: center;">
                <h2 style="color: {signal_color}; margin: 0;">{signal_info['signal']}</h2>
                <p><strong>Confidence:</strong> {signal_info['confidence']:.3f}</p>
                <p><strong>Sentiment Score:</strong> {signal_info['sentiment_score']:.3f}</p>
                <p><strong>Reason:</strong> {signal_info['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Stock Analysis - {selected_stock}")
        
        # Get stock data for selected symbol
        stock_data = st.session_state.final_data[
            st.session_state.final_data['symbol'] == selected_stock
        ].sort_values('Date')
        
        if not stock_data.empty:
            # Create comprehensive chart
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Stock Price & Moving Averages', 'RSI Indicator', 'Sentiment Score'),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price chart
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
            
            # Moving averages
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
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Sentiment
            if 'final_sentiment' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['final_sentiment'],
                        mode='lines+markers',
                        name='Sentiment',
                        line=dict(color='green'),
                        fill='tonexty'
                    ),
                    row=3, col=1
                )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"No chart data available for {selected_stock}")
    
    with col2:
        st.subheader("📈 All Signals")
        
        if 'signals' in st.session_state and not st.session_state.signals.empty:
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
                
                # Highlight selected stock
                border_width = "3px" if signal['symbol'] == selected_stock else "1px"
                
                st.markdown(f"""
                <div style="border: {border_width} solid {color}; border-radius: 8px; padding: 10px; margin: 8px 0; background-color: rgba(255,255,255,0.05);">
                    <h5 style="margin: 0;">{icon} {signal['symbol']}</h5>
                    <p style="margin: 2px 0;"><strong>{signal['signal']}</strong> | Conf: {signal['confidence']:.2f}</p>
                    <p style="margin: 2px 0; font-size: 0.8em;">Sentiment: {signal['sentiment_score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

elif st.session_state.data_loaded:
    st.info("✅ Data loaded successfully! Please select a stock from the dropdown in the sidebar.")

else:
    # Welcome screen
    st.info("👈 Click '🚀 Load Data & Train Model' in the sidebar to start analysis.")
    
    # Show sample data preview
    st.subheader("📋 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔄 Data Processing:**
        - Load news headlines from CSV
        - Extract stock tickers automatically
        - Download real-time price data
        - Calculate technical indicators
        
        **🤖 ML Models:**
        - LSTM for time series prediction
        - Random Forest for feature analysis
        - Hybrid ensemble approach
        """)
    
    with col2:
        st.markdown("""
        **📊 Trading Signals:**
        - 🟢 **BUY**: High confidence + positive sentiment
        - 🔴 **SELL**: High confidence + negative sentiment
        - 🟡 **HOLD**: Neutral or low confidence
        
        **📈 Features:**
        - Interactive charts and analysis
        - Real-time sentiment scoring
        - Technical indicator visualization
        """)
    
    # Show sample data if available
    try:
        st.subheader("📋 Sample Data Preview")
        data_loader = DataLoader(config)
        news_data = data_loader.load_news_data()
        
        if not news_data.empty:
            preview_data = news_data[['date', 'stock', 'headline', 'sentiment_label']].head(5)
            st.dataframe(preview_data, use_container_width=True)
            st.info(f"Found {len(news_data)} news articles for {news_data['stock'].nunique()} different stocks")
        else:
            st.warning("No sample data found. Please ensure 'data/sample_news_data.csv' exists.")
    except:
        st.warning("Unable to load sample data preview.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    💡 <strong>Sentiment Trading System</strong> | Educational purposes only<br>
    ⚠️ <em>Not financial advice. Please consult professionals before trading.</em>
</div>
""", unsafe_allow_html=True)
