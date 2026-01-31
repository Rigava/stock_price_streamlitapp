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
def compute_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Directional Movement
    up_move = high.diff()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing
    atr = tr.rolling(period).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    )

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return adx

RSI_OVERVALUED = 70
RSI_UNDERVALUE = 30

def classify_rsi(rsi):
    if rsi >= 70:
        return "Overvalued"
    elif rsi <= 30:
        return "Undervalued"
    else:
        return "Neutral"
def classify_regime(adx):
    if adx < 20:
        return "Range"
    elif adx < 25:
        return "Transition"
    else:
        return "Trend"

tickers = [
  "RELIANCE.NS", "LTF.NS","BEL.NS","JIOFIN.NS","COCHINSHIP.NS","HUDCO.NS","IREDA.NS","ADANIENT.NS","MOTHERSON.NS","NTPC.NS","IRCON.NS",
"ADANIGREEN.NS","IOC.NS","DOLATALGO.NS","NMDC.NS","MAHBANK.NS","RITES.NS","JSWINFRA.NS","IRFC.NS","VBL.NS","MARINE.NS","NCC.NS","IFCI.NS","RIBINFRA.NS"
]

# Streamlit app
# st.set_page_config(layout="wide")
# --- PAGE SETUP ---
st.set_page_config(page_title="JPN sailor", page_icon=":cop:",layout="wide")
st.title("NSE RSI Valuation Scanner-start small think big")

period = st.sidebar.selectbox("Timeframe", ["6mo", "1y", "2y"])
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

data = []

with st.spinner("Scanning JPN NSE stocks..."):
    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False)
        df.columns = df.columns.get_level_values(0)
        if df.empty:
            continue
        df['symbol'] = ticker
        df["RSI"] = compute_rsi(df["Close"], rsi_period)
        df["ADX"] = compute_adx(df)    
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df= df.dropna()
        df['%Change'] = ((df['Close'] / df['SMA_50'])-1)*100
        latest_rsi = df["RSI"].iloc[-1]
        latest_adx = df["ADX"].iloc[-1]
        latest_close = df['Close'].iloc[-1]
        latest_percent = df['%Change'].iloc[-1]
        
        

        data.append({
            "Ticker": ticker,
            "LTP": latest_close,
            "RSI": round(latest_rsi, 2),
            "Valuation": classify_rsi(latest_rsi),
            "Percent": latest_percent ,
            "ADX Trend": classify_regime(latest_adx)
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
