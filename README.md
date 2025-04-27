# Stock Portfolio Analyzer with AI Advisors

A Streamlit-based stock portfolio analyzer that provides current market data and AI-powered investment recommendations through dual agent perspectives.

## Features

- **Portfolio Overview**: View your current holdings, valuations, and performance metrics
- **Stock Analysis**: Analyze individual stocks with interactive charts and key metrics
- **AI Recommendations**: Get investment advice from two different AI perspectives:
  - **ValueMax**: Growth-focused agent with higher risk tolerance
  - **RiskGuard**: Conservative agent focused on capital preservation
- **AI Chat Interface**: Engage in direct conversations with both AI advisors
- **Indian Stock Support**: Optimized for Indian stock market (NSE) with .NS suffix

## Technology Stack

- **Streamlit**: For the interactive web interface
- **yfinance**: For fetching stock market data
- **Pandas**: For data manipulation and analysis
- **Plotly**: For interactive charts and visualizations
- **Matplotlib**: For technical analysis charts
- **Google Gemini AI**: For powering the AI advisors

## Usage

1. Upload your portfolio CSV file or use the provided sample
2. Navigate between different analysis sections
3. Consult the AI advisors for personalized recommendations
4. Engage in conversations with the AI agents for deeper insights

## Requirements

- Python 3.11+
- See requirements.txt or pyproject.toml for package dependencies

## Setup

```
pip install -r requirements.txt
streamlit run app.py
```

## License

[Choose an appropriate license]