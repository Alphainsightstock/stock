"""
Streamlit dashboard for AlphaInsights Sentiment Trading System
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.sentiment_analyzer import SentimentAnalyzer
from src.stock_data_processor import StockDataProcessor
from src.hybrid_model import HybridMLModel
from src.signal_generator import SignalGenerator
import config

# Page configuration
st.set_page_config(
    page_title="AlphaInsights - Sentiment Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.signal-buy {
    background: #10b981;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.signal-sell {
    background: #ef4444;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}

.signal-hold {
    background: #f59e0b;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

def load_data():
    """Load and process all data"""
    try:
        # Initialize components
        data_loader = DataLoader(config)
        sentiment_analyzer = SentimentAnalyzer()
        stock_processor = StockDataProcessor()
        hybrid_model = HybridMLModel(config)
        signal_generator = SignalGenerator(config)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load news data
        status_text.text('📰 Loading news data...')
        progress_bar.progress(10)
        news_data = data_loader.load_news_data()
        
        if news_data.empty:
            st.error("❌ No news data found! Please check the CSV file.")
            return None
        
        # Step 2: Analyze sentiment
        status_text.text('🧠 Analyzing sentiment...')
        progress_bar.progress(25)
        news_with_sentiment = sentiment_analyzer.analyze_news_sentiment(news_data)
        aggregated_sentiment = sentiment_analyzer.aggregate_stock_sentiment(news_with_sentiment)
        
        # Step 3: Load stock data
        status_text.text('📈 Loading stock price data...')
        progress_bar.progress(40)
        stock_data = data_loader.get_all_stock_data()
        
        if stock_data.empty:
            st.error("❌ No stock data found! Check internet connection.")
            return None
        
        # Step 4: Process technical indicators
        status_text.text('📊 Calculating technical indicators...')
        progress_bar.progress(60)
        stock_data_processed = stock_processor.calculate_technical_indicators(stock_data)
        
        # Step 5: Merge data
        status_text.text('🔗 Merging data...')
        progress_bar.progress(75)
        merged_data = stock_processor.merge_with_sentiment(
            stock_data_processed, aggregated_sentiment
        )
        final_data = stock_processor.prepare_features(merged_data)
        
        # Step 6: Train model
        status_text.text('🤖 Training ML model...')
        progress_bar.progress(85)
        final_data_scaled = stock_processor.scale_features(final_data)
        hybrid_model.train_model(final_data_scaled)
        
        # Step 7: Generate signals
        status_text.text('🎯 Generating trading signals...')
        progress_bar.progress(95)
        signals = signal_generator.generate_signals(final_data_scaled, hybrid_model)
        
        progress_bar.progress(100)
        status_text.text('✅ Complete!')
        
        return {
            'final_data': final_data,
            'signals': signals,
            'news_data': news_with_sentiment,
            'sentiment_summary': sentiment_analyzer.get_latest_sentiment_summary(aggregated_sentiment)
        }
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 AlphaInsights Sentiment Trading Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("*AI-powered sentiment analysis for Indian stock markets*")
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Load data button
    if st.sidebar.button("🚀 Load Data & Train Model", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True
        
        with st.spinner("Processing data... This may take a few minutes."):
            data = load_data()
            
            if data:
                st.session_state.update(data)
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
            
        st.session_state.processing = False
        st.rerun()
    
    # Show status
    if st.session_state.processing:
        st.sidebar.info("🔄 Processing...")
    
    if not st.session_state.data_loaded:
        st.info("👈 Click '🚀 Load Data & Train Model' in the sidebar to start analysis.")
        
        # Show system overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔄 Data Processing Pipeline
            - **Load News Data**: Process headlines from multiple sources
            - **Sentiment Analysis**: VADER sentiment with financial lexicon
            - **Stock Data**: Real-time price data from Yahoo Finance
            - **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 AI Trading Models
            - **Random Forest**: Ensemble learning for price direction
            - **Feature Engineering**: Technical + Sentiment indicators
            - **Signal Generation**: BUY/SELL/HOLD recommendations
            - **Risk Assessment**: Confidence scoring for each signal
            """)
        
        return
    
    # Stock selector
    available_stocks = []
    if 'final_data' in st.session_state and not st.session_state.final_data.empty:
        available_stocks = sorted(st.session_state.final_data['symbol'].unique())
    
    if available_stocks:
        selected_stock = st.sidebar.selectbox(
            "📊 Select Stock for Analysis",
            available_stocks,
            key="stock_selector"
        )
    else:
        selected_stock = None
        st.sidebar.warning("No stocks available")
    
    # Main content
    if st.session_state.data_loaded:
        
        # Trading signals summary
        st.subheader("🎯 Trading Signals Overview")
        
        if 'signals' in st.session_state and not st.session_state.signals.empty:
            signals_df = st.session_state.signals
            
            # Signal counts
            col1, col2, col3, col4 = st.columns(4)
            
            buy_count = len(signals_df[signals_df['signal'] == 'BUY'])
            sell_count = len(signals_df[signals_df['signal'] == 'SELL'])
            hold_count = len(signals_df[signals_df['signal'] == 'HOLD'])
            total_count = len(signals_df)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟢 BUY</h3>
                    <h2>{buy_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔴 SELL</h3>
                    <h2>{sell_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟡 HOLD</h3>
                    <h2>{hold_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 TOTAL</h3>
                    <h2>{total_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Signals distribution chart
            fig_pie = px.pie(
                values=[buy_count, sell_count, hold_count],
                names=['BUY', 'SELL', 'HOLD'],
                title="Trading Signals Distribution",
                color_discrete_map={'BUY': '#10b981', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Individual stock analysis
        if selected_stock:
            st.subheader(f"📊 Detailed Analysis - {selected_stock}")
            
            # Current signal for selected stock
            if 'signals' in st.session_state and not st.session_state.signals.empty:
                stock_signal = st.session_state.signals[
                    st.session_state.signals['symbol'] == selected_stock
                ]
                
                if not stock_signal.empty:
                    signal_info = stock_signal.iloc[0]
                    
                    # Signal display
                    signal_class = f"signal-{signal_info['signal'].lower()}"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        ### Current Signal
                        <div class="{signal_class}">
                            {signal_info['signal']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{signal_info['confidence']:.2f}")
                    
                    with col3:
                        st.metric("Sentiment Score", f"{signal_info['sentiment_score']:.3f}")
                    
                    st.markdown(f"**Reason:** {signal_info['reason']}")
            
            # Stock price chart with technical indicators
            if 'final_data' in st.session_state:
                stock_data = st.session_state.final_data[
                    st.session_state.final_data['symbol'] == selected_stock
                ].sort_values('Date')
                
                if not stock_data.empty:
                    # Create comprehensive chart
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=(
                            'Stock Price & Moving Averages',
                            'RSI Indicator',
                            'MACD',
                            'Sentiment Score'
                        ),
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.2, 0.2, 0.2]
                    )
                    
                    # Price and moving averages
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
                        
                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    if 'MACD' in stock_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=stock_data['Date'],
                                y=stock_data['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue')
                            ),
                            row=3, col=1
                        )
                    
                    if 'MACD_signal' in stock_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=stock_data['Date'],
                                y=stock_data['MACD_signal'],
                                mode='lines',
                                name='MACD Signal',
                                line=dict(color='red')
                            ),
                            row=3, col=1
                        )
                    
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
                            row=4, col=1
                        )
                    
                    fig.update_layout(
                        height=800,
                        title_text=f"{selected_stock} - Complete Technical Analysis",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # All signals table
        st.subheader("📋 All Trading Signals")
        
        if 'signals' in st.session_state and not st.session_state.signals.empty:
            # Format the signals table
            display_df = st.session_state.signals.copy()
            display_df['confidence'] = display_df['confidence'].round(3)
            display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
            display_df['ml_prediction'] = display_df['ml_prediction'].round(3)
            
            # Add color coding
            def highlight_signals(val):
                if val == 'BUY':
                    return 'background-color: #10b981; color: white'
                elif val == 'SELL':
                    return 'background-color: #ef4444; color: white'
                elif val == 'HOLD':
                    return 'background-color: #f59e0b; color: white'
                return ''
            
            styled_df = display_df.style.applymap(highlight_signals, subset=['signal'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Signals as CSV",
                data=csv,
                file_name=f"alphainsights_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
    <strong>AlphaInsights - Sentiment Trading System</strong><br>
    ⚠️ <em>This is for educational purposes only. Not financial advice. Please consult professionals before trading.</em><br>
    💡 Built with Streamlit • Powered by AI & Machine Learning
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
