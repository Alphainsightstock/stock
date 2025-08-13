"""
Streamlit dashboard for the sentiment trading project - Fixed dropdown issue and updated rerun
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
    for key in ['final_data', 'signals', 'news_data', 'sentiment_summary']:
        if key in st.session_state:
            del st.session_state[key]

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
            hybrid_model.train_model(final_data_scaled)

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

            # ✅ Updated: Force page refresh to update dropdown using st.rerun()
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.session_state.loading_in_progress = False
            st.session_state.data_loaded = False

# Display available stocks in sidebar
available_stocks = []
if st.session_state.data_loaded and 'final_data' in st.session_state:
    if not st.session_state.final_data.empty:
        available_stocks = sorted(st.session_state.final_data['symbol'].unique().tolist())
        selected_stock = st.sidebar.selectbox("📊 Select Stock for Analysis", available_stocks, key="stock_selector")
    else:
        selected_stock = st.sidebar.selectbox("📊 Select Stock for Analysis", ["No stocks available"], disabled=True)
else:
    selected_stock = st.sidebar.selectbox("📊 Select Stock for Analysis", ["Load data first..."], disabled=True)

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

        col1.metric("🟢 BUY Signals", buy_count)
        col2.metric("🔴 SELL Signals", sell_count)
        col3.metric("🟡 HOLD Signals", hold_count)

        # Show signal for selected stock
        stock_signal = st.session_state.signals[st.session_state.signals['symbol'] == selected_stock]
        if not stock_signal.empty:
            sig = stock_signal.iloc[0]
            color = "#00ff00" if sig['signal'] == 'BUY' else "#ff0000" if sig['signal'] == 'SELL' else "#ffaa00"
            st.markdown(f"""
            ### Current Signal for {selected_stock}
            <div style="border: 3px solid {color}; border-radius: 10px; padding: 20px; margin: 10px 0; text-align: center;">
                <h2 style="color: {color}; margin: 0;">{sig['signal']}</h2>
                <p><strong>Confidence:</strong> {sig['confidence']:.3f}</p>
                <p><strong>Sentiment Score:</strong> {sig['sentiment_score']:.3f}</p>
                <p><strong>Reason:</strong> {sig['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Main chart area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📊 Stock Analysis - {selected_stock}")
        stock_df = st.session_state.final_data[st.session_state.final_data['symbol'] == selected_stock].sort_values('Date')
        if not stock_df.empty:
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=('Stock Price & Moving Averages', 'RSI Indicator', 'Sentiment Score'),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)), row=1, col=1)
            if 'MA_10' in stock_df.columns:
                fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['MA_10'], mode='lines', name='MA 10', line=dict(color='orange', dash='dash')), row=1, col=1)
            if 'MA_30' in stock_df.columns:
                fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['MA_30'], mode='lines', name='MA 30', line=dict(color='red', dash='dot')), row=1, col=1)
            if 'RSI' in stock_df.columns:
                fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            if 'final_sentiment' in stock_df.columns:
                fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['final_sentiment'], mode='lines+markers', name='Sentiment', line=dict(color='green'), fill='tonexty'), row=3, col=1)
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 All Signals")
        if 'signals' in st.session_state and not st.session_state.signals.empty:
            for _, sig in st.session_state.signals.iterrows():
                c = "#00ff00" if sig['signal'] == 'BUY' else "#ff0000" if sig['signal'] == 'SELL' else "#ffaa00"
                bw = "3px" if sig['symbol'] == selected_stock else "1px"
                st.markdown(f"""
                <div style="border: {bw} solid {c}; border-radius: 8px; padding: 10px; margin: 8px 0;">
                    <h5>{sig['symbol']}</h5>
                    <p><strong>{sig['signal']}</strong> | Conf: {sig['confidence']:.2f}</p>
                    <p>Sentiment: {sig['sentiment_score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

elif st.session_state.data_loaded:
    st.info("✅ Data loaded successfully! Please select a stock in the sidebar.")
else:
    st.info("👈 Click '🚀 Load Data & Train Model' in the sidebar to start analysis.")
