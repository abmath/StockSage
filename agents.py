import os
import google.generativeai as genai
from portfolio_analysis import get_stock_price_history, get_stock_fundamentals, get_stock_info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Hardcoded API key
API_KEY = "AIzaSyC0rB8umNykwwRVIv8GOo06VP-v10h1qeA"

def setup_gemini():
    """Set up the Gemini API client"""
    try:
        genai.configure(api_key=API_KEY)
        return True
    except Exception as e:
        print(f"Error setting up Gemini: {e}")
        return False

def get_technical_analysis(ticker, period="1y"):
    """
    Generate technical analysis data for a stock
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for analysis
        
    Returns:
        dict: Technical analysis data and indicators
    """
    # Get historical data
    hist_data = get_stock_price_history(ticker, period=period)
    
    if hist_data.empty:
        return None
    
    # Calculate common technical indicators
    analysis = {}
    
    # Calculate moving averages
    hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = hist_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    hist_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    hist_data['EMA12'] = hist_data['Close'].ewm(span=12, adjust=False).mean()
    hist_data['EMA26'] = hist_data['Close'].ewm(span=26, adjust=False).mean()
    hist_data['MACD'] = hist_data['EMA12'] - hist_data['EMA26']
    hist_data['Signal'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Get the most recent values
    last_row = hist_data.iloc[-1]
    analysis['current_price'] = last_row['Close']
    analysis['ma20'] = last_row['MA20']
    analysis['ma50'] = last_row['MA50'] 
    analysis['ma200'] = last_row['MA200']
    analysis['rsi'] = last_row['RSI']
    analysis['macd'] = last_row['MACD']
    analysis['macd_signal'] = last_row['Signal']
    
    # Determine trend based on moving averages
    if last_row['Close'] > last_row['MA50'] > last_row['MA200']:
        analysis['trend'] = 'Bullish'
    elif last_row['Close'] < last_row['MA50'] < last_row['MA200']:
        analysis['trend'] = 'Bearish'
    else:
        analysis['trend'] = 'Mixed'
    
    # Calculate price changes over different periods
    analysis['change_1d'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-2] - 1) * 100
    analysis['change_1w'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-5] - 1) * 100 if len(hist_data) >= 5 else None
    analysis['change_1m'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-20] - 1) * 100 if len(hist_data) >= 20 else None
    analysis['change_3m'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-60] - 1) * 100 if len(hist_data) >= 60 else None
    
    # Calculate volatility (standard deviation of returns)
    returns = hist_data['Close'].pct_change().dropna()
    analysis['volatility'] = returns.std() * 100
    
    # Volume analysis
    analysis['avg_volume'] = hist_data['Volume'].mean()
    analysis['recent_volume'] = hist_data['Volume'].iloc[-5:].mean()
    analysis['volume_trend'] = 'Increasing' if analysis['recent_volume'] > analysis['avg_volume'] else 'Decreasing'
    
    # Store recent price data for chart generation
    analysis['recent_data'] = hist_data.iloc[-60:].copy()
    
    return analysis

def generate_chart_image(ticker, technical_data):
    """
    Generate a technical analysis chart image
    
    Args:
        ticker (str): Stock ticker symbol
        technical_data (dict): Technical analysis data
        
    Returns:
        str: Base64 encoded chart image
    """
    try:
        # Get recent data from technical analysis
        data = technical_data['recent_data']
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price and moving averages on top subplot
        ax1.plot(data.index, data['Close'], label='Price')
        ax1.plot(data.index, data['MA20'], label='MA20')
        ax1.plot(data.index, data['MA50'], label='MA50')
        ax1.plot(data.index, data['MA200'], label='MA200')
        ax1.set_title(f'{ticker} Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Volume in middle subplot
        ax2.bar(data.index, data['Volume'])
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # RSI in bottom subplot
        ax3.plot(data.index, data['RSI'], color='purple')
        ax3.axhline(y=70, color='red', linestyle='-')
        ax3.axhline(y=30, color='green', linestyle='-')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def get_reward_focused_agent_recommendation(ticker, portfolio_data=None):
    """
    Get recommendation from the reward-focused agent
    
    Args:
        ticker (str): Stock ticker symbol
        portfolio_data (DataFrame): Portfolio data
        
    Returns:
        dict: Recommendation with text and supporting data
    """
    try:
        setup_gemini()
        
        # Get technical analysis
        tech_analysis = get_technical_analysis(ticker)
        
        if not tech_analysis:
            return {
                "recommendation": f"Could not perform technical analysis for {ticker}.",
                "chart_image": None,
                "analysis_data": None
            }
        
        # Get fundamental data
        fundamentals = get_stock_fundamentals(ticker)
        basic_info = get_stock_info(ticker)
        
        # Generate chart image
        chart_image = generate_chart_image(ticker, tech_analysis)
        
        # Create a prompt for the reward-focused agent
        prompt = f"""
        You are ValueMax, an expert stock market analyst with a focus on maximizing returns and identifying high-growth opportunities. You are aggressive in seeking reward and have a high risk tolerance. Analyze the following data for {ticker} and provide a detailed recommendation.
        
        Technical Analysis Data:
        - Current Price: ₹{tech_analysis['current_price']:.2f}
        - Trend: {tech_analysis['trend']}
        - RSI (14-day): {tech_analysis['rsi']:.2f}
        - MACD: {tech_analysis['macd']:.2f}
        - Moving Averages: MA20=₹{tech_analysis['ma20']:.2f}, MA50=₹{tech_analysis['ma50']:.2f}, MA200=₹{tech_analysis['ma200']:.2f}
        - Recent Performance: 1D: {tech_analysis['change_1d']:.2f}%, 1W: {tech_analysis['change_1w']:.2f}%, 1M: {tech_analysis['change_1m']:.2f}%, 3M: {tech_analysis['change_3m']:.2f}%
        - Volatility: {tech_analysis['volatility']:.2f}%
        - Volume Trend: {tech_analysis['volume_trend']}
        
        Fundamental Data:
        {fundamentals}
        
        Based on this data, provide a detailed investment recommendation focusing on:
        1. Key technical signals and potential for short to medium-term gains
        2. Fundamental strengths that could drive growth
        3. Specific entry/exit points or price targets if applicable
        4. Growth catalysts and potential upside scenarios
        5. A clear buy/hold/sell recommendation with emphasis on potential reward
        
        Be specific, data-driven, and focus on maximizing returns. Format your response in a well-organized way with clear sections and bullet points where appropriate.
        """
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        response = model.generate_content(prompt)
        
        return {
            "recommendation": response.text,
            "chart_image": chart_image,
            "analysis_data": tech_analysis
        }
        
    except Exception as e:
        return {
            "recommendation": f"Error generating recommendation: {str(e)}",
            "chart_image": None,
            "analysis_data": None
        }

def get_risk_controller_agent_recommendation(ticker, portfolio_data=None):
    """
    Get recommendation from the risk-controller agent
    
    Args:
        ticker (str): Stock ticker symbol
        portfolio_data (DataFrame): Portfolio data
        
    Returns:
        dict: Recommendation with text and supporting data
    """
    try:
        setup_gemini()
        
        # Get technical analysis
        tech_analysis = get_technical_analysis(ticker)
        
        if not tech_analysis:
            return {
                "recommendation": f"Could not perform technical analysis for {ticker}.",
                "chart_image": None,
                "analysis_data": None
            }
        
        # Get fundamental data
        fundamentals = get_stock_fundamentals(ticker)
        basic_info = get_stock_info(ticker)
        
        # Generate chart image
        chart_image = generate_chart_image(ticker, tech_analysis)
        
        # Create a prompt for the risk-controller agent
        prompt = f"""
        You are RiskGuard, a cautious stock market analyst with a focus on capital preservation and risk management. You have a conservative approach to investing and prioritize protecting investments from downside risk. Analyze the following data for {ticker} and provide a detailed assessment with risk mitigation strategies.
        
        Technical Analysis Data:
        - Current Price: ₹{tech_analysis['current_price']:.2f}
        - Trend: {tech_analysis['trend']}
        - RSI (14-day): {tech_analysis['rsi']:.2f}
        - MACD: {tech_analysis['macd']:.2f}
        - Moving Averages: MA20=₹{tech_analysis['ma20']:.2f}, MA50=₹{tech_analysis['ma50']:.2f}, MA200=₹{tech_analysis['ma200']:.2f}
        - Recent Performance: 1D: {tech_analysis['change_1d']:.2f}%, 1W: {tech_analysis['change_1w']:.2f}%, 1M: {tech_analysis['change_1m']:.2f}%, 3M: {tech_analysis['change_3m']:.2f}%
        - Volatility: {tech_analysis['volatility']:.2f}%
        - Volume Trend: {tech_analysis['volume_trend']}
        
        Fundamental Data:
        {fundamentals}
        
        Based on this data, provide a detailed risk assessment focusing on:
        1. Key risk factors and technical warning signals
        2. Fundamental weaknesses or concerns
        3. Risk-adjusted return potential
        4. Potential defensive strategies (stop-loss levels, position sizing)
        5. Alternative lower-risk investments to consider
        6. A clear buy/hold/sell recommendation with emphasis on capital preservation
        
        Be specific, data-driven, and focus on identifying and mitigating potential risks. Format your response in a well-organized way with clear sections and bullet points where appropriate.
        """
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        response = model.generate_content(prompt)
        
        return {
            "recommendation": response.text,
            "chart_image": chart_image,
            "analysis_data": tech_analysis
        }
        
    except Exception as e:
        return {
            "recommendation": f"Error generating recommendation: {str(e)}",
            "chart_image": None,
            "analysis_data": None
        }