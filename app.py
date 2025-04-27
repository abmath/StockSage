import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from io import StringIO
import os
import time

from portfolio_analysis import (
    calculate_portfolio_metrics,
    get_stock_fundamentals,
    get_stock_info,
    get_stock_price_history
)
from gemini_integration import get_gemini_recommendation
from utils import format_currency, format_percentage, validate_portfolio_data

# Page configuration
st.set_page_config(
    page_title="Stock Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

# Define app header
def display_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Stock Portfolio Analyzer")
        st.markdown("Upload your portfolio data and get AI-powered recommendations")
    with col2:
        st.image("https://images.unsplash.com/photo-1508589066756-c5dfb2cb7587", width=200)

# Sidebar for app navigation and settings
def display_sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Portfolio Overview", "Stock Analysis", "AI Recommendations"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application analyzes your stock portfolio, "
        "provides fundamental data, and offers AI-powered "
        "investment recommendations using Google's Gemini AI."
    )
    
    st.sidebar.markdown("---")
    
    # Display random stock market image in sidebar
    image_urls = [
        "https://images.unsplash.com/photo-1554260570-e9689a3418b8",
        "https://images.unsplash.com/photo-1488459716781-31db52582fe9",
        "https://images.unsplash.com/photo-1744782211816-c5224434614f",
        "https://images.unsplash.com/photo-1444653614773-995cb1ef9efa",
        "https://images.unsplash.com/photo-1563986768711-b3bde3dc821e"
    ]
    
    import random
    random_img = random.choice(image_urls)
    st.sidebar.image(random_img, use_column_width=True)
    
    return page

# Portfolio uploader
def upload_portfolio():
    st.markdown("### Upload Your Portfolio")
    uploaded_file = st.file_uploader("Upload your portfolio CSV file", type=["csv"])
    
    sample_data = None
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            df = pd.read_csv(StringIO(content))
            
            # Validate the required columns
            validation_result = validate_portfolio_data(df)
            if validation_result['valid']:
                sample_data = df
                st.success("Portfolio data loaded successfully!")
            else:
                st.error(f"Error: {validation_result['message']}")
                st.markdown("Your CSV should contain the following columns:")
                st.markdown("- Instrument: Stock ticker/symbol")
                st.markdown("- Qty: Number of shares")
                st.markdown("- Avg cost: Average cost per share")
                st.markdown("- Other optional columns: LTP, Invested, Cur val, P&L, Net chg, Day chg")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Download sample template
    col1, col2 = st.columns(2)
    
    with col1:
        # If user wants to use the example
        if st.checkbox("Use example portfolio (Indian stocks)"):
            sample_data = pd.DataFrame({
                "Instrument": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS"],
                "Qty": [10, 15, 25, 20, 40],
                "Avg cost": [2450.75, 3280.50, 1600.00, 1320.50, 790.75],
                "LTP": [None, None, None, None, None],
                "Invested": [None, None, None, None, None],
                "Cur val": [None, None, None, None, None],
                "P&L": [None, None, None, None, None],
                "Net chg": [None, None, None, None, None],
                "Day chg": [None, None, None, None, None]
            })
            st.success("Example portfolio with Indian stocks loaded!")
    
    with col2:
        # Create downloadable sample template
        template_data = pd.DataFrame({
            "Instrument": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
            "Qty": [10, 5, 20],
            "Avg cost": [2450.75, 3280.50, 1600.00]
        })
        
        @st.cache_data
        def convert_df_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df_to_csv(template_data)
        
        st.download_button(
            label="Download Sample Template (CSV)",
            data=csv,
            file_name='portfolio_template.csv',
            mime='text/csv',
            help="Download this template, fill it with your portfolio data, and upload it above"
        )
    
    return sample_data

# Portfolio overview page
def display_portfolio_overview(portfolio_data):
    st.markdown("## Portfolio Overview")
    
    if portfolio_data is None:
        st.info("Please upload your portfolio data to see analysis")
        return
    
    with st.spinner("Fetching current market data..."):
        # Calculate key metrics
        updated_portfolio = calculate_portfolio_metrics(portfolio_data)
        
        # Display key metrics
        total_invested = updated_portfolio["Invested"].sum()
        current_value = updated_portfolio["Cur val"].sum()
        total_pl = updated_portfolio["P&L"].sum()
        pl_percentage = (total_pl / total_invested * 100) if total_invested > 0 else 0
        
        # Display top level metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        
        with col2:
            st.metric("Current Value", f"₹{current_value:,.2f}")
        
        with col3:
            st.metric("Total P&L", f"₹{total_pl:,.2f}", f"{pl_percentage:.2f}%")
        
        with col4:
            # Count profitable vs non-profitable positions
            profitable = (updated_portfolio["P&L"] > 0).sum()
            total = len(updated_portfolio)
            st.metric("Profitable Positions", f"{profitable}/{total}", f"{profitable/total*100:.1f}%")
        
        # Display portfolio table
        st.markdown("### Portfolio Details")
        display_df = updated_portfolio.copy()
        
        # Format currency and percentage columns
        display_cols = {
            "Avg cost": format_currency,
            "LTP": format_currency,
            "Invested": format_currency,
            "Cur val": format_currency,
            "P&L": format_currency,
            "Net chg": format_percentage,
            "Day chg": format_percentage
        }
        
        for col, formatter in display_cols.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(formatter)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Portfolio composition chart
        st.markdown("### Portfolio Composition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for allocation
            fig = px.pie(
                updated_portfolio,
                values="Cur val",
                names="Instrument",
                title="Portfolio Allocation",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for P&L
            fig = px.bar(
                updated_portfolio,
                x="Instrument",
                y="P&L",
                title="Profit & Loss by Position",
                color="P&L",
                color_continuous_scale=["red", "green"],
                text="P&L"
            )
            fig.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance chart over time
        st.markdown("### Recent Portfolio Performance")
        
        # Get historical data for portfolio stocks
        try:
            # Get 6 months of historical data for each stock
            tickers = updated_portfolio["Instrument"].tolist()
            historical_data = {}
            
            for ticker in tickers:
                historical_data[ticker] = get_stock_price_history(ticker, period="6mo")
            
            # Create portfolio performance chart
            if historical_data:
                # Combine all historical data into one chart
                fig = go.Figure()
                
                for ticker in tickers:
                    if ticker in historical_data and not historical_data[ticker].empty:
                        df = historical_data[ticker]
                        fig.add_trace(
                            go.Scatter(
                                x=df.index, 
                                y=df['Close'], 
                                name=ticker,
                                mode='lines'
                            )
                        )
                
                fig.update_layout(
                    title="6-Month Price History",
                    xaxis_title="Date",
                    yaxis_title="Stock Price (₹)",
                    legend_title="Stocks",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not retrieve historical data for your portfolio stocks.")
        except Exception as e:
            st.error(f"Error retrieving historical data: {e}")

# Individual stock analysis page
def display_stock_analysis(portfolio_data):
    st.markdown("## Stock Analysis")
    
    if portfolio_data is None:
        st.info("Please upload your portfolio data to see analysis")
        return
    
    # Let the user select a stock from their portfolio
    stocks = portfolio_data["Instrument"].unique().tolist()
    
    # Allow user to analyze a stock not in portfolio
    custom_ticker = st.text_input("Enter a stock ticker to analyze (or select from your portfolio below):")
    if custom_ticker:
        selected_stock = custom_ticker.upper()
    else:
        selected_stock = st.selectbox("Select a stock from your portfolio:", stocks)
    
    if not selected_stock:
        st.warning("Please select a stock to analyze")
        return
    
    with st.spinner(f"Analyzing {selected_stock}..."):
        try:
            # Get stock info
            info = get_stock_info(selected_stock)
            
            if not info:
                st.error(f"Could not retrieve data for {selected_stock}. Please check the ticker symbol.")
                return
            
            # Display stock info
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if 'logo_url' in info and info['logo_url']:
                    st.image(info['logo_url'])
                st.subheader(info.get('shortName', selected_stock))
                st.caption(f"{info.get('exchange', '')}: {selected_stock}")
                
                current_price = info.get('currentPrice', 0)
                previous_close = info.get('previousClose', 0)
                price_change = current_price - previous_close
                price_change_percent = (price_change / previous_close * 100) if previous_close else 0
                
                price_color = "green" if price_change >= 0 else "red"
                st.markdown(f"<h3 style='color: {price_color};'>₹{current_price:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {price_color};'>{price_change:.2f} ({price_change_percent:.2f}%)</p>", unsafe_allow_html=True)
            
            with col2:
                # Display key stats
                st.markdown("### Key Statistics")
                
                # Create columns for metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Market Cap", format_currency(info.get('marketCap', 0)))
                    st.metric("52 Week High", format_currency(info.get('fiftyTwoWeekHigh', 0)))
                    st.metric("Volume", f"{info.get('volume', 0):,}")
                
                with metric_col2:
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                    st.metric("52 Week Low", format_currency(info.get('fiftyTwoWeekLow', 0)))
                    st.metric("Avg Volume", f"{info.get('averageVolume', 0):,}")
                
                with metric_col3:
                    st.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
                    st.metric("Target Price", format_currency(info.get('targetMeanPrice', 0)))
                    st.metric("Beta", f"{info.get('beta', 0):.2f}")
            
            # Display price chart
            st.markdown("### Price History")
            
            # Time period selector
            periods = {
                "1 Month": "1mo",
                "3 Months": "3mo",
                "6 Months": "6mo",
                "YTD": "ytd",
                "1 Year": "1y",
                "5 Years": "5y"
            }
            selected_period = st.select_slider("Select time period:", options=list(periods.keys()))
            
            # Get price history
            hist_data = get_stock_price_history(selected_stock, period=periods[selected_period])
            
            if not hist_data.empty:
                # Plot interactive price chart
                fig = go.Figure()
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name="Price"
                    )
                )
                
                # Add volume as bar chart on secondary y-axis
                fig.add_trace(
                    go.Bar(
                        x=hist_data.index,
                        y=hist_data['Volume'],
                        name="Volume",
                        yaxis="y2",
                        opacity=0.3
                    )
                )
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title=f"{selected_stock} Price History - {selected_period}",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No historical data available for {selected_stock}")
            
            # Fundamentals
            st.markdown("### Fundamentals")
            
            fundamentals = get_stock_fundamentals(selected_stock)
            
            if fundamentals:
                fund_col1, fund_col2 = st.columns(2)
                
                with fund_col1:
                    st.subheader("Income Statement Highlights")
                    st.metric("Revenue", format_currency(fundamentals.get('totalRevenue', 0)))
                    st.metric("Gross Profit", format_currency(fundamentals.get('grossProfit', 0)))
                    st.metric("Net Income", format_currency(fundamentals.get('netIncome', 0)))
                    st.metric("EPS", f"₹{fundamentals.get('eps', 0):.2f}")
                
                with fund_col2:
                    st.subheader("Balance Sheet Highlights")
                    st.metric("Total Assets", format_currency(fundamentals.get('totalAssets', 0)))
                    st.metric("Total Debt", format_currency(fundamentals.get('totalDebt', 0)))
                    st.metric("Cash", format_currency(fundamentals.get('totalCash', 0)))
                    st.metric("Debt-to-Equity", f"{fundamentals.get('debtToEquity', 0):.2f}")
            else:
                st.warning(f"Fundamental data not available for {selected_stock}")
            
            # Analyst recommendations
            st.markdown("### Analyst Recommendations")
            
            if 'recommendationMean' in info:
                rec_value = info.get('recommendationMean', 3)
                rec_text = "N/A"
                
                if 1 <= rec_value < 1.5:
                    rec_text = "Strong Buy"
                elif 1.5 <= rec_value < 2.5:
                    rec_text = "Buy"
                elif 2.5 <= rec_value < 3.5:
                    rec_text = "Hold"
                elif 3.5 <= rec_value < 4.5:
                    rec_text = "Sell"
                elif rec_value >= 4.5:
                    rec_text = "Strong Sell"
                
                st.metric("Analyst Rating", rec_text, f"{rec_value:.2f}/5")
                
                # Create a visual indicator
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = rec_value,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Analyst Consensus"},
                    gauge = {
                        'axis': {'range': [1, 5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [1, 1.5], 'color': "green"},
                            {'range': [1.5, 2.5], 'color': "lightgreen"},
                            {'range': [2.5, 3.5], 'color': "yellow"},
                            {'range': [3.5, 4.5], 'color': "orange"},
                            {'range': [4.5, 5], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': rec_value
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Analyst recommendations not available for this stock")
            
        except Exception as e:
            st.error(f"Error analyzing stock: {e}")

# AI recommendations page
def display_ai_recommendations(portfolio_data):
    st.markdown("## AI-Powered Recommendations")
    
    if portfolio_data is None:
        st.info("Please upload your portfolio data to see AI recommendations")
        return
    
    # Import the agents module
    from agents import get_reward_focused_agent_recommendation, get_risk_controller_agent_recommendation
    
    st.markdown("""
    Our dual AI agents provide investment advice from two different perspectives:
    
    - **ValueMax**: Maximizes returns with higher risk tolerance
    - **RiskGuard**: Preserves capital with conservative risk management
    """)
    
    st.warning("AI recommendations are for informational purposes only and should not be considered financial advice.")
    
    # Prepare portfolio for analysis
    updated_portfolio = calculate_portfolio_metrics(portfolio_data)
    
    # Let user select analysis type
    analysis_type = st.radio(
        "Select analysis type:",
        ["AI Investment Chat", "Portfolio Overview", "Individual Stock Analysis", "Rebalancing Suggestions"]
    )
    
    # Initialize session state for chat histories if they don't exist
    if "valuemax_chat_history" not in st.session_state:
        st.session_state.valuemax_chat_history = []
    
    if "riskguard_chat_history" not in st.session_state:
        st.session_state.riskguard_chat_history = []
    
    # AI Advisor Chat section (now the primary option)
    if analysis_type == "AI Investment Chat":
        st.markdown("### Chat with our AI Investment Advisors")
        
        # Create a layout with two columns for the chat interfaces
        chat_col1, chat_col2 = st.columns(2)
        
        with chat_col1:
            st.markdown("### ValueMax (Growth-Focused)")
            st.markdown("*Focused on maximizing returns with higher risk tolerance*")
            
            # Show ValueMax chat history
            for message in st.session_state.valuemax_chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                else:
                    st.markdown(f"**ValueMax**: {message['content']}")
            
            # ValueMax chat input
            valuemax_question = st.text_input("Ask ValueMax about your investments:", key="valuemax_input")
            
            if st.button("Ask ValueMax") and valuemax_question:
                with st.spinner("ValueMax is thinking..."):
                    try:
                        # Add user question to ValueMax chat history
                        st.session_state.valuemax_chat_history.append({
                            "role": "user",
                            "content": valuemax_question
                        })
                        
                        # Create ValueMax prompt
                        value_max_prompt = f"""
                        You are ValueMax, an aggressive investment advisor who focuses on maximizing returns and identifying high-growth opportunities.
                        You have a high risk tolerance and are willing to accept volatility for greater gains.
                        
                        A user has asked the following question about their investments:
                        "{valuemax_question}"
                        
                        Previous conversation context:
                        {str(st.session_state.valuemax_chat_history[:-1]) if len(st.session_state.valuemax_chat_history) > 1 else "This is the first question."}
                        
                        Provide a helpful response from your perspective as ValueMax. Make your advice specific, actionable,
                        and aligned with your philosophy of pursuing strong returns even if it means higher risk. 
                        Keep your response under 200 words and use a confident, growth-oriented tone.
                        """
                        
                        # Get response from ValueMax
                        value_max_response = get_gemini_recommendation(value_max_prompt)
                        
                        # Add response to ValueMax chat history
                        st.session_state.valuemax_chat_history.append({
                            "role": "assistant",
                            "content": value_max_response
                        })
                        
                        # Force a rerun to show the updated chat
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error getting ValueMax advice: {e}")
        
        with chat_col2:
            st.markdown("### RiskGuard (Protection-Focused)")
            st.markdown("*Focused on preserving capital with conservative approach*")
            
            # Show RiskGuard chat history
            for message in st.session_state.riskguard_chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                else:
                    st.markdown(f"**RiskGuard**: {message['content']}")
            
            # RiskGuard chat input
            riskguard_question = st.text_input("Ask RiskGuard about your investments:", key="riskguard_input")
            
            if st.button("Ask RiskGuard") and riskguard_question:
                with st.spinner("RiskGuard is thinking..."):
                    try:
                        # Add user question to RiskGuard chat history
                        st.session_state.riskguard_chat_history.append({
                            "role": "user",
                            "content": riskguard_question
                        })
                        
                        # Create RiskGuard prompt
                        risk_guard_prompt = f"""
                        You are RiskGuard, a conservative investment advisor who focuses on capital preservation and risk management.
                        You prioritize stable investments with lower volatility and are cautious about potential downside risks.
                        
                        A user has asked the following question about their investments:
                        "{riskguard_question}"
                        
                        Previous conversation context:
                        {str(st.session_state.riskguard_chat_history[:-1]) if len(st.session_state.riskguard_chat_history) > 1 else "This is the first question."}
                        
                        Provide a helpful response from your perspective as RiskGuard. Make your advice specific, actionable,
                        and aligned with your philosophy of protecting capital and managing risk carefully.
                        Keep your response under 200 words and use a measured, prudent tone.
                        """
                        
                        # Get response from RiskGuard
                        risk_guard_response = get_gemini_recommendation(risk_guard_prompt)
                        
                        # Add response to RiskGuard chat history
                        st.session_state.riskguard_chat_history.append({
                            "role": "assistant",
                            "content": risk_guard_response
                        })
                        
                        # Force a rerun to show the updated chat
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error getting RiskGuard advice: {e}")
        
        # Add a feature to ask both agents at once
        st.markdown("### Ask Both Advisors")
        dual_question = st.text_input("Ask a question to both advisors:", key="dual_input")
        
        if st.button("Consult Both Advisors") and dual_question:
            with st.spinner("Consulting both advisors..."):
                try:
                    # Add user question to both chat histories
                    st.session_state.valuemax_chat_history.append({
                        "role": "user",
                        "content": dual_question
                    })
                    
                    st.session_state.riskguard_chat_history.append({
                        "role": "user",
                        "content": dual_question
                    })
                    
                    # Create prompts for both agents
                    value_max_prompt = f"""
                    You are ValueMax, an aggressive investment advisor who focuses on maximizing returns and identifying high-growth opportunities.
                    You have a high risk tolerance and are willing to accept volatility for greater gains.
                    
                    A user has asked the following question about their investments:
                    "{dual_question}"
                    
                    Previous conversation context:
                    {str(st.session_state.valuemax_chat_history[:-1]) if len(st.session_state.valuemax_chat_history) > 1 else "This is the first question."}
                    
                    Provide a helpful response from your perspective as ValueMax. Make your advice specific, actionable,
                    and aligned with your philosophy of pursuing strong returns even if it means higher risk. 
                    Keep your response under 200 words and use a confident, growth-oriented tone.
                    """
                    
                    risk_guard_prompt = f"""
                    You are RiskGuard, a conservative investment advisor who focuses on capital preservation and risk management.
                    You prioritize stable investments with lower volatility and are cautious about potential downside risks.
                    
                    A user has asked the following question about their investments:
                    "{dual_question}"
                    
                    Previous conversation context:
                    {str(st.session_state.riskguard_chat_history[:-1]) if len(st.session_state.riskguard_chat_history) > 1 else "This is the first question."}
                    
                    Provide a helpful response from your perspective as RiskGuard. Make your advice specific, actionable,
                    and aligned with your philosophy of protecting capital and managing risk carefully.
                    Keep your response under 200 words and use a measured, prudent tone.
                    """
                    
                    # Get responses from both agents
                    value_max_response = get_gemini_recommendation(value_max_prompt)
                    risk_guard_response = get_gemini_recommendation(risk_guard_prompt)
                    
                    # Add responses to respective chat histories
                    st.session_state.valuemax_chat_history.append({
                        "role": "assistant",
                        "content": value_max_response
                    })
                    
                    st.session_state.riskguard_chat_history.append({
                        "role": "assistant",
                        "content": risk_guard_response
                    })
                    
                    # Force a rerun to show the updated chats
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error getting advice: {e}")
                    st.error("Please check your API key configuration or try again later.")
    
    # Individual Stock Analysis section
    elif analysis_type == "Individual Stock Analysis":
        # Let user choose which stock to analyze
        stock_to_analyze = st.selectbox(
            "Select a stock for detailed analysis:",
            updated_portfolio["Instrument"].tolist()
        )
        
        # Added option to view both agents' perspectives
        view_option = st.radio(
            "View recommendations from:",
            ["ValueMax Only", "RiskGuard Only", "Both Agents Side-by-Side"]
        )
        
        if st.button("Generate AI Recommendation"):
            with st.spinner(f"Consulting our AI advisors about {stock_to_analyze}..."):
                try:
                    # Get recommendations from agents based on the view option
                    if view_option == "Both Agents Side-by-Side":
                        # Get both recommendations
                        valuemax_result = get_reward_focused_agent_recommendation(stock_to_analyze, updated_portfolio)
                        riskguard_result = get_risk_controller_agent_recommendation(stock_to_analyze, updated_portfolio)
                        
                        # Display the technical analysis chart if available (only need to show once)
                        if valuemax_result["chart_image"]:
                            st.image(f"data:image/png;base64,{valuemax_result['chart_image']}", caption=f"{stock_to_analyze} Technical Analysis Chart", use_container_width=True)
                        
                        # Display technical indicators in a collapsible section
                        if valuemax_result["analysis_data"]:
                            with st.expander("View Technical Indicators"):
                                tech_data = valuemax_result["analysis_data"]
                                
                                # Create columns for metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Price", f"${tech_data['current_price']:.2f}")
                                    st.metric("RSI (14-day)", f"{tech_data['rsi']:.2f}")
                                    st.metric("Trend", tech_data['trend'])
                                
                                with col2:
                                    st.metric("MA20", f"${tech_data['ma20']:.2f}")
                                    st.metric("MA50", f"${tech_data['ma50']:.2f}")
                                    st.metric("MA200", f"${tech_data['ma200']:.2f}")
                                
                                with col3:
                                    st.metric("1-Day Change", f"{tech_data['change_1d']:.2f}%")
                                    st.metric("1-Month Change", f"{tech_data['change_1m']:.2f}%" if tech_data['change_1m'] else "N/A")
                                    st.metric("Volatility", f"{tech_data['volatility']:.2f}%")
                        
                        # Create columns for side-by-side recommendations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### ValueMax Recommendation")
                            st.info(valuemax_result["recommendation"])
                        
                        with col2:
                            st.markdown("### RiskGuard Recommendation")
                            st.info(riskguard_result["recommendation"])
                    else:
                        # Get single agent recommendation
                        if "ValueMax" in view_option:
                            result = get_reward_focused_agent_recommendation(stock_to_analyze, updated_portfolio)
                            agent_name = "ValueMax"
                        else:
                            result = get_risk_controller_agent_recommendation(stock_to_analyze, updated_portfolio)
                            agent_name = "RiskGuard"
                        
                        # Display recommendation
                        st.markdown(f"### {agent_name} Recommendation")
                        
                        # Display the technical analysis chart if available
                        if result["chart_image"]:
                            st.image(f"data:image/png;base64,{result['chart_image']}", caption=f"{stock_to_analyze} Technical Analysis Chart")
                        
                        # Display the recommendation text
                        st.info(result["recommendation"])
                        
                        # Display technical indicators in a collapsible section
                        if result["analysis_data"]:
                            with st.expander("View Technical Indicators"):
                                tech_data = result["analysis_data"]
                                
                                # Create columns for metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Price", f"${tech_data['current_price']:.2f}")
                                    st.metric("RSI (14-day)", f"{tech_data['rsi']:.2f}")
                                    st.metric("Trend", tech_data['trend'])
                                
                                with col2:
                                    st.metric("MA20", f"${tech_data['ma20']:.2f}")
                                    st.metric("MA50", f"${tech_data['ma50']:.2f}")
                                    st.metric("MA200", f"${tech_data['ma200']:.2f}")
                                
                                with col3:
                                    st.metric("1-Day Change", f"{tech_data['change_1d']:.2f}%")
                                    st.metric("1-Month Change", f"{tech_data['change_1m']:.2f}%" if tech_data['change_1m'] else "N/A")
                                    st.metric("Volatility", f"{tech_data['volatility']:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error generating AI recommendation: {e}")
                    st.error("Please check your API key configuration or try again later.")
    
    # Portfolio Overview section - show both perspectives
    elif analysis_type == "Portfolio Overview":
        # Added option to view both agents' perspectives
        view_option = st.radio(
            "View recommendations from:",
            ["ValueMax Only", "RiskGuard Only", "Both Agents Side-by-Side"]
        )
        
        if st.button("Generate AI Recommendation"):
            with st.spinner("Analyzing your entire portfolio with AI..."):
                try:
                    # Prepare portfolio data summary
                    portfolio_summary = updated_portfolio.to_dict('records')
                    
                    if view_option == "Both Agents Side-by-Side":
                        # ValueMax perspective
                        valuemax_prompt = f"""
                        As ValueMax, a financial analyst with a focus on maximizing returns and high-growth opportunities,
                        provide an overall assessment of this investment portfolio:
                        
                        {portfolio_summary}
                        
                        Include:
                        1. Overall portfolio health and diversification
                        2. Key strengths and growth opportunities
                        3. Sector exposure analysis
                        4. Growth opportunities and return potential
                        5. General recommendations for improvement focused on maximizing returns
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # RiskGuard perspective
                        riskguard_prompt = f"""
                        As RiskGuard, a financial analyst with a focus on capital preservation and risk management,
                        provide an overall assessment of this investment portfolio:
                        
                        {portfolio_summary}
                        
                        Include:
                        1. Overall portfolio health and diversification
                        2. Key risk factors and potential vulnerabilities
                        3. Sector exposure analysis
                        4. Risk assessment and downside protection strategies
                        5. General recommendations for improvement focused on risk reduction
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # Get recommendations from both perspectives
                        valuemax_recommendation = get_gemini_recommendation(valuemax_prompt)
                        riskguard_recommendation = get_gemini_recommendation(riskguard_prompt)
                        
                        # Create columns for side-by-side recommendations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### ValueMax Portfolio Analysis")
                            st.info(valuemax_recommendation)
                        
                        with col2:
                            st.markdown("### RiskGuard Portfolio Analysis")
                            st.info(riskguard_recommendation)
                    else:
                        # Single agent perspective
                        if "ValueMax" in view_option:
                            agent_perspective = "focus on maximizing returns and high-growth opportunities"
                            growth_focus = "Growth opportunities and return potential"
                            agent_name = "ValueMax"
                        else:
                            agent_perspective = "focus on capital preservation and risk management"
                            growth_focus = "Risk assessment and downside protection"
                            agent_name = "RiskGuard"
                        
                        prompt = f"""
                        As a financial analyst with a {agent_perspective}, provide an overall assessment of this investment portfolio:
                        
                        {portfolio_summary}
                        
                        Include:
                        1. Overall portfolio health and diversification
                        2. Key strengths and weaknesses
                        3. Sector exposure analysis
                        4. {growth_focus}
                        5. General recommendations for improvement based on your investment philosophy
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # Get recommendation from Gemini
                        recommendation = get_gemini_recommendation(prompt)
                        
                        # Display recommendation in an expanded box
                        st.markdown(f"### {agent_name} Portfolio Analysis")
                        st.info(recommendation)
                    
                except Exception as e:
                    st.error(f"Error generating AI recommendation: {e}")
                    st.error("Please check your API key configuration or try again later.")
    
    # Rebalancing Suggestions section
    else:  # Rebalancing Suggestions
        # Added option to view both agents' perspectives
        view_option = st.radio(
            "View recommendations from:",
            ["ValueMax Only", "RiskGuard Only", "Both Agents Side-by-Side"]
        )
        
        if st.button("Generate AI Recommendation"):
            with st.spinner("Analyzing portfolio allocation with AI..."):
                try:
                    # Prepare portfolio data for rebalancing analysis
                    portfolio_summary = updated_portfolio.to_dict('records')
                    
                    # Calculate current allocations by stock
                    total_value = updated_portfolio["Cur val"].sum()
                    allocations = []
                    for idx, row in updated_portfolio.iterrows():
                        allocations.append({
                            'Instrument': row['Instrument'],
                            'Allocation': f"{(row['Cur val'] / total_value * 100):.2f}%"
                        })
                    
                    if view_option == "Both Agents Side-by-Side":
                        # ValueMax perspective
                        valuemax_prompt = f"""
                        As ValueMax, a financial advisor with a focus on maximizing returns and growth potential,
                        suggest portfolio rebalancing opportunities based on this data:
                        
                        Current Portfolio:
                        {portfolio_summary}
                        
                        Current Allocations:
                        {allocations}
                        
                        Include:
                        1. Analysis of current allocation and its growth potential
                        2. Opportunities to increase exposure to high-growth assets
                        3. Specific rebalancing recommendations focused on enhancing returns
                        4. Target allocation suggestions aligned with maximizing growth
                        5. Potential high-upside investments to consider
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # RiskGuard perspective
                        riskguard_prompt = f"""
                        As RiskGuard, a financial advisor with a focus on reducing risk and preserving capital,
                        suggest portfolio rebalancing opportunities based on this data:
                        
                        Current Portfolio:
                        {portfolio_summary}
                        
                        Current Allocations:
                        {allocations}
                        
                        Include:
                        1. Analysis of current allocation and its risk profile
                        2. Potential overexposure to volatile or risky assets
                        3. Specific rebalancing recommendations focused on reducing volatility and downside risk
                        4. Target allocation suggestions aligned with capital preservation
                        5. Defensive assets to consider for portfolio protection
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # Get recommendations from both perspectives
                        valuemax_recommendation = get_gemini_recommendation(valuemax_prompt)
                        riskguard_recommendation = get_gemini_recommendation(riskguard_prompt)
                        
                        # Create columns for side-by-side recommendations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### ValueMax Rebalancing Strategy")
                            st.info(valuemax_recommendation)
                        
                        with col2:
                            st.markdown("### RiskGuard Rebalancing Strategy")
                            st.info(riskguard_recommendation)
                    else:
                        # Single agent perspective
                        if "ValueMax" in view_option:
                            agent_perspective = "focus on maximizing returns and growth potential"
                            growth_or_risk = "growth potential"
                            exposure_focus = "Opportunities to increase exposure to high-growth assets"
                            rebalance_goal = "enhancing returns"
                            investment_type = "Potential high-upside investments to consider"
                            agent_name = "ValueMax"
                        else:
                            agent_perspective = "focus on reducing risk and preserving capital"
                            growth_or_risk = "risk profile"
                            exposure_focus = "Potential overexposure to volatile or risky assets"
                            rebalance_goal = "reducing volatility and downside risk"
                            investment_type = "Defensive assets to consider for portfolio protection"
                            agent_name = "RiskGuard"
                        
                        prompt = f"""
                        As a financial advisor with a {agent_perspective}, suggest portfolio rebalancing opportunities based on this data:
                        
                        Current Portfolio:
                        {portfolio_summary}
                        
                        Current Allocations:
                        {allocations}
                        
                        Include:
                        1. Analysis of current allocation and its {growth_or_risk}
                        2. {exposure_focus}
                        3. Specific rebalancing recommendations focused on {rebalance_goal}
                        4. Target allocation suggestions aligned with your investment philosophy
                        5. {investment_type}
                        
                        Format your response in a well-organized way with clear sections and bullet points where appropriate.
                        """
                        
                        # Get recommendation from Gemini
                        recommendation = get_gemini_recommendation(prompt)
                        
                        # Display recommendation in an expanded box
                        st.markdown(f"### {agent_name} Rebalancing Strategy")
                        st.info(recommendation)
                    
                except Exception as e:
                    st.error(f"Error generating AI recommendation: {e}")
                    st.error("Please check your API key configuration or try again later.")

# Main application
def main():
    display_header()
    page = display_sidebar()
    
    # Upload portfolio data
    portfolio_data = upload_portfolio()
    
    # Display selected page
    if page == "Portfolio Overview":
        display_portfolio_overview(portfolio_data)
    elif page == "Stock Analysis":
        display_stock_analysis(portfolio_data)
    else:  # AI Recommendations
        display_ai_recommendations(portfolio_data)

if __name__ == "__main__":
    main()
