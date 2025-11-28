import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache_resource
def fit_sarima(df, order=(1,1,1), seasonal_order=(1,1,1,7), column="sales"):
    """
    Fit a SARIMA model on the given series.
    Parameters:
        df (pd.DataFrame): DataFrame with 'sales' column
        order (tuple): ARIMA order (p,d,q)
        seasonal_order (tuple): Seasonal order (P,D,Q,s)
        column (str): column name to forecast
    Returns:
        Fitted SARIMA results object
    """
    model = SARIMAX(df[column], order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)

def forecast(model, steps=30):
    """Generate forecast from fitted SARIMA model."""
    return model.forecast(steps=steps)