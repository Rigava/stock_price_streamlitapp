import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import ta
import plotly.graph_objects as go

# Technical Indicators
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
    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_20"],
        name="EMA 20"
    ))
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

# Streamlit app
def main():
    st.title("📈 AI-Powered Technical Analysis")
    symbol_list = ["RELIANCE", "SBIN","TCS","INFY","ITC"]
    symbol = st.selectbox("Select stock symbol", symbol_list)
    if symbol:
        ticker= symbol.upper() + ".NS"
        try:
            nifty_data = yf.download(tickers=ticker, period="5y")
            nifty_data.columns = nifty_data.columns.get_level_values(0)
            latest_price = nifty_data['Close'].iloc[-1]
            st.success(f"The latest price is: {latest_price}")
            # Plotting historical price movement
            
            # image = plot_chart(df)
            if st.button("Analyze"):
                df = add_indicators(nifty_data)
                st.plotly_chart(plot_chart(df), use_container_width=True)            
                summary = chart_summary(df)
                # ai_response = analyze_with_llm(summary)
                st.subheader("📊 Recommendation")
                st.markdown(summary)
            # Export data as CSV
            st.subheader("Export Data")
            if st.button("Export as CSV"):
                st.write("Exporting stock data as CSV...")
                df.to_csv(f"{symbol}_data.csv", index=False)
                st.success("Stock data exported successfully!")
        except Exception as e:
            st.error("Error occurred while fetching stock data.")
            st.error(e)


# Run the app
if __name__ == '__main__':
    main()
