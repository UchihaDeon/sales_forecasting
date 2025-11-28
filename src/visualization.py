import matplotlib.pyplot as plt
import streamlit as st

def plot_model_overlays(df, forecasts: dict, title="Forecast Comparison"):
    """
    Plot actual sales with multiple model forecasts overlayed.
    forecasts: dict with keys as model names and values as forecast Series
    """
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['sales'], label="Actual Sales", color="black")

    for model_name, forecast_series in forecasts.items():
        plt.plot(forecast_series.index, forecast_series.values, label=f"{model_name} Forecast")

    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    st.pyplot(plt)