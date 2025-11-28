import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

@st.cache_resource
def fit_arima(df, order=(1,1,1), column="sales"):
    model = ARIMA(df[column], order=order)
    return model.fit()

def forecast(model, steps=30):
    return model.forecast(steps=steps)