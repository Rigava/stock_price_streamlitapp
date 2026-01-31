import yfinance as yf
import pandas as pd
import numpy as np

# RSI engine
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
# ADX Regime
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
