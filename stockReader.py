import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit app
def main():
    st.title("Nifty Stocks Price Update")
    # symbol = st.text_input("Enter stock symbol (e.g., SBIN, RELIANCE)")
    symbol_list = ["RELIANCE", "SBIN","TCS","INFY","ITC"]
    symbol = st.selectbox("Select stock symbol", symbol_list)

    if symbol:

        ticker= symbol.upper() + ".NS"
        try:
            nifty_data = yf.download(tickers=ticker, period="5y")
            latest_price = nifty_data['Close'].iloc[-1]
            st.success(f"The latest price is: {latest_price}")
            # Plotting historical price movement
            st.dataframe(nifty_data)

            # Export data as CSV
            st.subheader("Export Data")
            if st.button("Export as CSV"):
                st.write("Exporting stock data as CSV...")
                nifty_data.to_csv(f"{symbol}_data.csv", index=False)
                st.success("Stock data exported successfully!")
        except Exception as e:
            st.error("Error occurred while fetching stock data.")
            st.error(e)


# Run the app
if __name__ == '__main__':
    main()
