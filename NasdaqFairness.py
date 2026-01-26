import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# rsi engine
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
  
RSI_OVERVALUED = 70
RSI_UNDERVALUE = 30

def classify_rsi(rsi):
    if rsi >= 70:
        return "Overvalued"
    elif rsi <= 30:
        return "Undervalued"
    else:
        return "Neutral"
NASDAQ_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "NFLX", "AMD", "INTC"
]
tickers = [
    "ACHR","INTC","ARM","NVDA","TSLA","MSFT","GOOGL","AAPL","AMZN",
    "META","AVGO","DELL","DEO","AMD","FLY","COIN","ASML","JOBY",
    "LULU","PLTR","NIO","ISRG","CRWD","KO","BRK-B","WMT","CRM"
]

# Streamlit app
# st.set_page_config(layout="wide")
# --- PAGE SETUP ---
st.set_page_config(page_title="BIG B sailor", page_icon=":cop:",layout="wide")
st.title("NASDAQ RSI Valuation Scanner-start small think big")

period = st.sidebar.selectbox("Timeframe", ["6mo", "1y", "2y"])
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

data = []

with st.spinner("Scanning Big B NASDAQ stocks..."):
    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False)

        if df.empty:
            continue

        df["RSI"] = compute_rsi(df["Close"], rsi_period)
        latest_rsi = df["RSI"].iloc[-1]
        latest_close = df['Close'].iloc[-1].values[0]
        

        data.append({
            "Ticker": ticker,
            "LTP": latest_close,
            "RSI": round(latest_rsi, 2),
            "Valuation": classify_rsi(latest_rsi)
        })

result_df = pd.DataFrame(data)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔴 Overvalued")
    st.dataframe(
        result_df[result_df["Valuation"] == "Overvalued"]
        .sort_values("RSI", ascending=False)
    )

with col2:
    st.subheader("⚪ Neutral")
    st.dataframe(
        result_df[result_df["Valuation"] == "Neutral"]
    )

with col3:
    st.subheader("🟢 Undervalued")
    st.dataframe(
        result_df[result_df["Valuation"] == "Undervalued"]
        .sort_values("RSI")
    )
