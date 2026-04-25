import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import ta
import plotly.graph_objects as go
import json
import pandas as pd
import numpy as np
import time
#From utilityffunction
from utilityFunction import compute_adx
#LLM Config
from langchain_google_genai import ChatGoogleGenerativeAI
api_key = st.secrets.API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]
tickers_list = tickers.SYMBOL.to_list()
tickers_list= tickers_list[5:]
tickers_list.remove("TATAMOTORS")
symbol_list=[]
for symbols in tickers_list:
    s= symbols.upper() + ".NS"
    symbol_list.append(s)
symbol_list.append("^NSEI")
# Technical Indicators utilities helper functions
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["EMA_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["MACD_SIGNAL"] = ta.trend.MACD(df["Close"]).macd_signal()
    return df
# Plotly Charts
def plot_chart(df):
    fig = go.Figure()
    fig.add_candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
    fig.add_trace(go.Scatter(x=df.index,y=df["EMA_20"],name="EMA 20"))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        name="EMA 50"
    ))
    fig.update_layout(height=600)
    return fig
def chart_summary(df):
    latest = df.iloc[-1]
    prev = df.iloc[-5:]

    return f"""
    Latest:
    Close: {latest['Close']}
    RSI: {latest['RSI']}
    EMA20: {latest['EMA_20']}
    EMA50: {latest['EMA_50']}
    MACD: {latest['MACD']}
    MACD Signal: {latest['MACD_SIGNAL']}

    Trend Context:
    - EMA20 Trend (5 periods): {prev['EMA_20'].is_monotonic_increasing}
    - EMA50 Trend (5 periods): {prev['EMA_50'].is_monotonic_increasing}
    - Price Momentum: {prev['Close'].iloc[-1] - prev['Close'].iloc[0]}

    Momentum Context:
    - RSI Direction: {"Rising" if prev['RSI'].iloc[-1] > prev['RSI'].iloc[0] else "Falling"}
    - MACD Crossover Recent: {prev['MACD'].iloc[-1] > prev['MACD_SIGNAL'].iloc[-1]}

    Structure:
    - Recent High: {df['Close'].tail(20).max()}
    - Recent Low: {df['Close'].tail(20).min()}
    """
def get_analysis(chart_summary, symbol):
    
    prompt = f"""
    You are an institutional-grade quantitative trader and technical analyst.
    
    Your goal is NOT just to describe the market, but to generate a high-quality, executable trade decision with risk management and probabilistic thinking.

    Symbol: {symbol}
    
    Technical Summary:
    {chart_summary}
    
    -----------------------------------
    INSTRUCTIONS
    -----------------------------------
    
    Follow this structured reasoning process:
    
    1. MARKET REGIME DETECTION
    Classify the current market condition:
    - Trending Up / Trending Down / Sideways / High Volatility
    
    Use EMA structure, RSI behavior, and MACD momentum.
    
    2. SIGNAL QUALITY SCORING
    Score the trade setup from 1 to 10 based on:
    - Indicator alignment (EMA, RSI, MACD)
    - Momentum strength
    - Noise / false signal risk
    
    3. INDICATOR CONFLICT ANALYSIS
    - Identify any contradictions between indicators
    - Resolve which signal has priority
    - If conflict is high → reduce confidence or avoid trade
    
    4. TRADE DECISION
    Provide one clear action:
    - Buy / Sell / Hold / Avoid
    
    Avoid forcing trades if setup is weak.
    
    5. ENTRY STRATEGY
    Define optimal entry:
    - Immediate / Pullback / Breakout
    
    Provide exact trigger condition (not vague).
    
    6. RISK MANAGEMENT (MANDATORY)
    Define:
    - Entry price (reference latest close if needed)
    - Stop Loss (logical, based on structure)
    - Targets (at least 2 levels)
    
    Ensure:
    - Minimum Risk:Reward = 1:2
    - If not achievable → mark trade as "Avoid"
    
    7. PROBABILITY & EDGE
    Estimate:
    - Win probability (%)
    - Type of edge:
      (Momentum / Mean Reversion / Breakout)
    
    8. SCENARIO PLANNING
    Define 3 scenarios:
    - Bullish continuation
    - Bearish reversal
    - Sideways movement
    
    Each must include:
    - Trigger
    - Action
    
    9. TRADE FILTER
    Explicitly decide:
    - Should this trade be taken?
    
    Reject if:
    - Weak momentum
    - Indicator conflict
    - Poor RR

    Provide the output in a valid JSON format with below fields and without escaped characters and formatting artifacts.

    """
    response = llm.invoke(prompt)  # just pass plain string to LLM
    decoded_content = json.dumps(response.content)
    dict_resp = json.loads(decoded_content)
    return dict_resp  # <-- fix: .content
# Helper functions
def extract_json_object(text):
    start = text.find('{')
    end =  text.rfind('}')
    json_text = text[start:end+1]
    return json_text
# -----------------------------------------Main functon-------------------------------------------------- #
def main():
    st.title("📈 AI-Powered Technical analyst post")
    symbol = st.selectbox("Select stock symbol", symbol_list)
    try:
        nifty_data = yf.download(tickers=symbol, period="5y")
        nifty_data.columns = nifty_data.columns.get_level_values(0)
        latest_price = nifty_data['Close'].iloc[-1]
        st.success(f"The latest price of {symbol} is: {latest_price}")
        with st.expander("Show data"):
            st.dataframe(nifty_data)
        if st.button("Analyze"):
            # st.write(symbol)
            # nifty_data = yf.download(tickers=symbol, period="5y")
            # nifty_data.columns = nifty_data.columns.get_level_values(0)
            # latest_price = nifty_data['Close'].iloc[-1]
            df = add_indicators(nifty_data)
            with st.expander("Show data"):
                st.dataframe(df)
            st.plotly_chart(plot_chart(df), use_container_width=True) 
            st.spinner("Analysing")
            summary = chart_summary(df)
            ai_response = get_analysis(summary,symbol)
            json_str = extract_json_object(ai_response)
            data = json.loads(json_str)
              
            st.subheader("📊 Recommendation")
            st.write(data)
            # st.info(data['Action'],icon="ℹ️")
            # st.info(data['Justification'],icon="✅")
            # st.warning(data['Trade plan'],icon="⚠️")
            # Export data as CSV
        if st.button("Export as CSV"):
             st.write("Exporting stock data as CSV...")
             df.to_csv(f"{symbol}_data.csv", index=False)
             st.success("Stock data exported successfully!")
        st.snow()
    except Exception as e:
      st.error("Error occurred while fetching stock data.")
      st.error(e)


# Run the app
if __name__ == '__main__':
    main()
