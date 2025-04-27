import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time

def get_stock_info(ticker):
    """
    Get basic information about a stock
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Stock information or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return None

def get_stock_price_history(ticker, period="1mo", interval="1d"):
    """
    Get historical price data for a stock
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        DataFrame: Historical price data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_fundamentals(ticker):
    """
    Get fundamental financial data for a stock
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Fundamental data or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Combine various fundamental data points
        fundamentals = {}
        
        # Balance sheet data
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            # Get most recent period
            latest_bs = balance_sheet.iloc[:, 0]
            
            fundamentals['totalAssets'] = latest_bs.get('Total Assets', np.nan)
            fundamentals['totalLiabilities'] = latest_bs.get('Total Liabilities Net Minority Interest', np.nan)
            fundamentals['totalEquity'] = latest_bs.get('Total Equity Gross Minority Interest', np.nan)
            fundamentals['totalCash'] = latest_bs.get('Cash And Cash Equivalents', np.nan)
            fundamentals['totalDebt'] = latest_bs.get('Total Debt', np.nan)
            
            # Calculate debt-to-equity ratio
            if fundamentals.get('totalEquity') and fundamentals.get('totalDebt'):
                fundamentals['debtToEquity'] = fundamentals['totalDebt'] / fundamentals['totalEquity']
        
        # Income statement data
        income_stmt = stock.income_stmt
        if not income_stmt.empty:
            # Get most recent period
            latest_is = income_stmt.iloc[:, 0]
            
            fundamentals['totalRevenue'] = latest_is.get('Total Revenue', np.nan)
            fundamentals['grossProfit'] = latest_is.get('Gross Profit', np.nan)
            fundamentals['operatingIncome'] = latest_is.get('Operating Income', np.nan)
            fundamentals['netIncome'] = latest_is.get('Net Income', np.nan)
            fundamentals['eps'] = latest_is.get('Diluted EPS', np.nan)
        
        # Cash flow data
        cash_flow = stock.cashflow
        if not cash_flow.empty:
            # Get most recent period
            latest_cf = cash_flow.iloc[:, 0]
            
            fundamentals['operatingCashFlow'] = latest_cf.get('Operating Cash Flow', np.nan)
            fundamentals['freeCashFlow'] = latest_cf.get('Free Cash Flow', np.nan)
        
        # Add key ratios from info
        info = stock.info
        for key in ['trailingPE', 'forwardPE', 'priceToBook', 'dividendYield', 'returnOnEquity', 'debtToEquity']:
            if key in info:
                fundamentals[key] = info[key]
        
        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None

def get_current_stock_price(ticker):
    """
    Get the current stock price for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Current stock price or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return current_price
        
        return None
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
        return None

def calculate_portfolio_metrics(portfolio_df):
    """
    Calculate portfolio metrics including current values and P&L
    
    Args:
        portfolio_df (DataFrame): Portfolio data with Instrument, Qty, and Avg cost columns
        
    Returns:
        DataFrame: Updated portfolio with calculated metrics
    """
    # Make a copy to avoid modifying the original
    df = portfolio_df.copy()
    
    # Ensure numeric columns are numeric
    numeric_cols = ["Qty", "Avg cost", "LTP", "Invested", "Cur val", "P&L", "Net chg", "Day chg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # For each stock, get current price and calculate metrics
    for idx, row in df.iterrows():
        ticker = row["Instrument"]
        qty = row["Qty"]
        avg_cost = row["Avg cost"]
        
        # Fetch current price
        current_price = get_current_stock_price(ticker)
        
        if current_price is not None:
            # Calculate metrics
            invested = qty * avg_cost
            current_value = qty * current_price
            pl = current_value - invested
            net_change = (pl / invested) * 100 if invested > 0 else 0
            
            # Get day change percentage from yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                day_change = info.get('regularMarketChangePercent', 0)
            except:
                day_change = 0
            
            # Update the dataframe
            df.at[idx, "LTP"] = current_price
            df.at[idx, "Invested"] = invested
            df.at[idx, "Cur val"] = current_value
            df.at[idx, "P&L"] = pl
            df.at[idx, "Net chg"] = net_change
            df.at[idx, "Day chg"] = day_change
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.1)
    
    return df
