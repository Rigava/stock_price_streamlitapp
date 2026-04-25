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

    return f"""
    Latest Close: {latest['Close']}
    RSI: {latest['RSI']}
    EMA 20: {latest['EMA_20']}
    EMA 50: {latest['EMA_50']}
    MACD: {latest['MACD']}
    MACD Signal: {latest['MACD_SIGNAL']}

    Trend:
    - EMA 20 above EMA 50: {latest['EMA_20'] > latest['EMA_50']}
    - RSI Overbought (>70): {latest['RSI'] > 70}
    - RSI Oversold (<30): {latest['RSI'] < 30}
    """
def get_analysis(chart_summary,symbol):
  prompt = f"""
  You are a stock trader specializing in technical analysis at a top financial institution.
  Provide a short summary of Industry fundamentals and overall macro economic factors affecting the {symbol}.
  Based on short summary and the following technical indicators, provide:
  1.Recommendation: Buy / Sell / Hold
  2.Justification in simple language
  3.Best trading plan based on the support and resistance with risk and reward information

  Technical data:
  {chart_summary}

  Provide the output in a valid JSON format with below fields and without escaped characters and formatting artifacts.
  if the sample text does not contain enough information to provide the below fields return none but do not
  create any information on your own.
  '''
  {{
  "Action": "...",
  "Justification":"...",
  "Trade plan": "...."
  }}'''
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
    if symbol:
        
        try:
            nifty_data = yf.download(tickers=symbol, period="5y")
            nifty_data.columns = nifty_data.columns.get_level_values(0)
            latest_price = nifty_data['Close'].iloc[-1]
            st.success(f"The latest price of {symbol} is: {latest_price}")
            if st.button("Analyze"):
                df = add_indicators(nifty_data)
                with st.expander("Show data"):
                    st.dataframe(df)
                # st.plotly_chart(plot_chart(df), use_container_width=True)            
                summary = chart_summary(df)
                ai_response = get_analysis(summary,symbol)
                json_str = extract_json_object(ai_response)
                data = json.loads(json_str)
                  
                st.subheader("📊 Recommendation")
                st.info(data['Action'],icon="ℹ️")
                st.info(data['Justification'],icon="✅")
                st.warning(data['Trade plan'],icon="⚠️")
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
