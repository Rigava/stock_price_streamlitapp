import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit app
def main():
    st.title("Nifty Stocks Price Update")
    # symbol = st.text_input("Enter stock symbol (e.g., SBIN, RELIANCE)")
    symbol_list = ["RELIANCE", "SBIN","TCS","INFY"]
    symbol = st.selectbox("Select stock symbol", symbol_list)

    if symbol:

        ticker= symbol.upper() + ".NS"
        try:
            nifty_data = yf.download(tickers=ticker, start="2020-01-01", end="2023-05-18")
            latest_price = nifty_data['Close'].iloc[-1]
            st.success(f"The latest price is: {latest_price}")
            # Plotting historical price movement
            st.subheader("Historical Price Movement")
            plt.figure(figsize=(10, 6))
            plt.plot(nifty_data.index, nifty_data['Close'])
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Price Movement')
            plt.xticks(rotation=45)
            st.pyplot(plt)
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
