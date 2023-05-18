import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from urllib.parse import quote

# Streamlit app
def main():
    symbol_list = ["RELIANCE", "SBIN","TCS","INFY","HDFC","ITC"]
    symbol = st.selectbox("Select stock symbol", symbol_list)
    encoded_symbol=quote(symbol)

    st.title("Nifty Stocks Price Update")
    # symbol = st.text_input("Enter stock symbol (e.g., SBIN, RELIANCE)")

    if symbol:

        try:
            stock_url='https://www.nseindia.com/api/historical/cm/equity?symbol={}'.format(encoded_symbol)
            headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36' ,
            "accept-encoding": "gzip, deflate, br", "accept-language": "en-US,en;q=0.9"}
            r = requests.get(stock_url, headers=headers).json()
            data_values=[data for data in r['data']]
            nifty_data=pd.DataFrame(data_values)
            latest_price = nifty_data['CH_CLOSING_PRICE'].iloc[-1]
            st.success(f"The latest price is: {latest_price}")
            # Plotting historical price movement
            st.subheader("Historical Price Movement")
            plt.figure(figsize=(10, 6))
            plt.plot(nifty_data.index, nifty_data['CH_CLOSING_PRICE'])
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
