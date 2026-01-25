import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import ta
import plotly.graph_objects as go
import json
import pandas as pd

#LLM Config
from langchain_google_genai import ChatGoogleGenerativeAI
api_key = st.secrets.API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Technical Indicators
def add_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["EMA_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["MACD_SIGNAL"] = ta.trend.MACD(df["Close"]).macd_signal()
    return df
    
def MACDIndicator(df):
    df['EMA12']= df.Close.ewm(span=12).mean()
    df['EMA26']= df.Close.ewm(span=26).mean()
    df['MACD'] = df.EMA12 - df.EMA26
    df['Signal'] = df.MACD.ewm(span=9).mean()
    df['MACD_diff']=df.MACD - df.Signal
    df.loc[(df['MACD_diff']>0) & (df.MACD_diff.shift(1)<0),'Decision MACD']='Buy'
    df.loc[(df['MACD_diff']<0) & (df.MACD_diff.shift(1)>0),'Decision MACD']='Sell'
    df.dropna()
    print('MACD indicators added')
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
#LLM Prompt
def get_analysis(chart_summary,symbol):

  prompt = f"""
  You are a stock trader specializing in technical analysis at a top financial institution.
  Provide a short summary of Industry fundamentals and overall macro economic factors affecting the {symbol}.
  Based on short summary and the following technical indicators, provide:
  1.Recommendation: Buy / Sell / Hold
  2.Justification in simple language
  3.Risk factors

  Technical data:
  {chart_summary}

  Provide the output in a valid JSON format with below fields and without escaped characters and formatting artifacts.
  if the sample text does not contain enough information to provide the below fields return none but do not
  create any information on your own.
  '''
  {{
  "Summary::"...",
  "Action": "...",
  "Justification":"...",
  "Risk": "...."
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
def get_value_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    balance = stock.balance_sheet
    income = stock.financials
    ratio={}
    try:
        ratio["Ticker"] = ticker,
        ratio["Market Cap"] = info.get("marketCap")
        ratio["PE"] = info.get("trailingPE")
        ratio["PB"] = info.get("priceToBook")
        ratio["EV_EBITDA"] = info.get("enterpriseToEbitda")
        ratio["ROE"] = info.get("returnOnEquity")
        ratio["Debt_Equity"] = info.get("debtToEquity")
        ratio["Current_Ratio"] = info.get("currentRatio")
        ratio["Operating_Margin"] = info.get("operatingMargins")
        ratio["FCF"] = info.get("freeCashflow")

        netIn = income.loc["Net Income"][0]
        equity = balance.loc['Stockholders Equity'][0]
        debt =  balance.loc['Total Debt'][0]
        ratio["Equity"] = balance.loc['Stockholders Equity'][0]
        ratio["Debt"] = balance.loc['Total Debt'][0]
        ratio["Net Income"] = income.loc["Net Income"][0]
        ratio["d_Debt_Equity"] = round(debt / equity, 2)
        ratio["d_ROE"] = round((netIn / equity)*100,2)
    except Exception:
        pass
    return ratio
    
def value_score(row):
    score = 0

    if row["PE"] and row["PE"] < 15:
        score += 1
    if row["PB"] and row["PB"] < 1.5:
        score += 1
    if row["EV_EBITDA"] and row["EV_EBITDA"] < 10:
        score += 1
    if row["d_Debt_Equity"] and row["d_Debt_Equity"] < 1:
        score += 1
    if row["Current_Ratio"] and row["Current_Ratio"] > 1.5:
        score += 1
    if row["d_ROE"] and row["d_ROE"] > 15:
        score += 1
    if row["Operating_Margin"] and row["Operating_Margin"] > 0.15:
        score += 1
    if row["FCF"] and row["FCF"] > 0:
        score += 1

    return score
# Streamlit app
def main():
    st.title("📈 AI-Powered Technical Analysis")
    # symbol_list = ["RELIANCE", "SBIN","TCS","INFY","ITC"]
    tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]
   
    symbol_list = tickers.SYMBOL.to_list()
    symbol_list= symbol_list[5:]
    symbol_list.remove("TATAMOTORS")

    #SHORTLIST FEATURE
    shortlist_option = st.sidebar.selectbox("select strategy",["MACD","Value","Growth","RSI","Breakout"])
    if st.button("Shortlist", use_container_width=True):
        Buy = []
        Sell = []
        Hold = []
        framelist = [] # add OHLC data
        data =[] # add fundamental data
        for stock in symbol_list:
            yf_tick = stock.upper()+".NS"
            df = yf.download(tickers=yf_tick, period="1y")
            df.columns = df.columns.get_level_values(0)
            df = MACDIndicator(df)
            framelist.append(df)
            #Fetch fundamentals
            data.append( get_value_fundamentals(yf_tick))


            # Determine buy or sell recommendation based on last two rows of the data to provide buy & sell signals
            if shortlist_option=="MACD":                
                if df['Decision MACD'].iloc[-1]=='Buy':    
                    Buy.append(stock)
                elif df['Decision MACD'].iloc[-1]=='Sell':
                    Sell.append(stock)
                else:
                    Hold.append(stock)  
        df_funda = pd.DataFrame(data)

        df_funda["Value Score"] = df_funda.apply(value_score, axis=1)
        with st.expander("Show the Fundamentals",expanded = False):
            st.dataframe(df_funda[['Ticker','Net Income','Equity','Debt','PE','PB','ROE','Debt_Equity','Current_Ratio','EV_EBITDA','FCF','Value Score']])
        if shortlist_option=="Value": 
            filter_buy = df_funda[df_funda['Value Score'] > 4]
            Buy = filter_buy['Ticker'].tolist()
            filter_sell = df_funda[df_funda['Value Score'] < 2]
            Sell = filter_sell['Ticker'].tolist()

        # Display stock data and recommendation
        st.write(":blue[List of stock with positive signal]")
        st.table({"Stocks":Buy})
        st.write(":blue[List of stock with negative signal]")
        st.table({"Stocks":Sell})
    
    symbol = st.selectbox("Select stock symbol", symbol_list)
    if symbol:
        ticker= symbol.upper() + ".NS"
        try:
            nifty_data = yf.download(tickers=ticker, period="5y")
            nifty_data.columns = nifty_data.columns.get_level_values(0)
            latest_price = nifty_data['Close'].iloc[-1]
            st.success(f"The latest price of {symbol} is: {latest_price}")

            if st.button("Analyze"):
                df = add_indicators(nifty_data)
                st.plotly_chart(plot_chart(df), use_container_width=True)            
                summary = chart_summary(df)
                ai_response = get_analysis(summary,symbol)
                json_str = extract_json_object(ai_response)
                data = json.loads(json_str)
                
                st.subheader("📊 Recommendation")
                st.info(data['Summary'],icon="ℹ️")
                st.info(data['Action'],icon="ℹ️")
                st.info(data['Justification'],icon="✅")
                st.warning(data['Risk'],icon="⚠️")
               
            # Export data as CSV
            st.subheader("Export Data")
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
