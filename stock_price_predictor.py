import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
 
st.title("STOCK PRICE PREDICTOR")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days') # MA STANDS FOR MOVING AVERAGE
st.text('ORIGINAL PRICE IN BLUE')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
st.text('ORIGINAL PRICE IN BLUE')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
st.text('ORIGINAL PRICE IN BLUE')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.text('ORIGINAL PRICE IN BLUE AND MA FOR 100 DAYS IN YELLOW AND MA FOR 250 DAYS IN GREEN')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
disclaimer = """
DISCLAIMER:

The stock predictions provided are based on the analysis of historical stock data using various data science techniques and methodologies. While efforts have been made to ensure accuracy, stock markets are inherently volatile and unpredictable. Past performance is not indicative of future results.

Important Points to Note:
1. No Guarantee of Accuracy: The predictions are generated using historical data and may not account for all variables that can impact stock prices. Unexpected market events, changes in economic conditions, and other unforeseen factors can significantly affect stock prices.
2. Use at Your Own Risk: Any investment decisions made based on these predictions are done at your own risk. It is essential to conduct your own research and consider your financial situation, investment goals, and risk tolerance before making any investment decisions.
3. Seek Professional Advice: It is highly recommended to consult with a licensed financial advisor or professional before making any investment decisions. They can provide personalized advice tailored to your specific circumstances.
4. Regular Updates: Stock markets can change rapidly. It is crucial to stay informed about current market conditions and regularly update your analysis to make well-informed decisions.
5. Educational Purposes: This prediction is provided for educational and informational purposes only. It should not be considered as financial or investment advice.

Conclusion:
Investing in stocks involves risks, including the loss of principal. Make sure to diversify your portfolio and invest wisely. Always stay informed and consider seeking professional guidance when needed.
"""

st.text(disclaimer)