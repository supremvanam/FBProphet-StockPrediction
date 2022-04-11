import streamlit as st
from datetime import date
import styles

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(
     page_title="COMP 377 | Group 6",
    #  layout="wide",   
 )
st.title("Predictr - Stock Prediction App")



hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_style, unsafe_allow_html=True)
st.markdown(styles.footer,unsafe_allow_html=True)


stocks = ("AAPL", "GOOG", "AMZN", "MSFT", "GME", "NFLX")
# selected_stock = st.selectbox("Select dataset for prediction", stocks)
selected_stock = st.text_input("Select dataset for prediction", placeholder="Example: AAPL")


n_years = st.slider("Years of prediction", 1, 5)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

if(selected_stock):
    data = load_data(selected_stock)
    st.subheader("Raw data")
    st.write(data.tail())
    plot_raw_data()

    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    with st.spinner(f'Predicting "{selected_stock}" stock for {n_years} year(s)'):
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())


        st.write('forecast data')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write('forecast components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)
    st.success(f'Stock prediction for {selected_stock} is succesful')
