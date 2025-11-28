import streamlit as st
import pandas as pd
from src.preprocessing import load_data, clean_data
from src.evaluation import mape, rmse
from src.visualization import plot_actual_vs_forecast, plot_model_overlays

st.set_page_config(page_title="Sales Forecasting System", layout="wide")
st.title("📈 Sales Forecasting System")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Forecast", "Evaluation", "Visualization"])

# ---------------- Dataset Tab ----------------
with tab1:
    st.header("Upload & Preview Dataset")
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
        st.write("Data Preview", df.head())
        st.session_state['df'] = df
    else:
        st.info("Upload a CSV with 'date' and 'sales' columns to begin.")

# ---------------- Forecast Tab ----------------
with tab2:
    st.header("Run Forecasts")
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Shared forecast horizon slider
        steps = st.slider("Forecast horizon (days)", 7, 90, 30, key="horizon_slider")

        # Unified model choice
        model_choice = st.selectbox("Choose forecasting model", ["ARIMA", "SARIMA", "LSTM"], key="model_choice")

        # --- ARIMA ---
        if model_choice == "ARIMA":
            order_p = st.number_input("ARIMA p", 0, 5, 1, key="arima_p")
            order_d = st.number_input("ARIMA d", 0, 2, 1, key="arima_d")
            order_q = st.number_input("ARIMA q", 0, 5, 1, key="arima_q")

            if st.button("Run ARIMA Forecast", key="run_arima"):
                from src.models.arima_model import fit_arima, forecast
                res = fit_arima(df, order=(order_p, order_d, order_q))
                yhat = forecast(res, steps)
                future_index = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=steps, freq='D')
                forecast_series = pd.Series(yhat.values, index=future_index)
                st.session_state['forecast_arima'] = forecast_series
                plot_actual_vs_forecast(df, forecast_series, title="ARIMA Forecast")

        # --- SARIMA ---
        elif model_choice == "SARIMA":
            seasonal_p = st.number_input("Seasonal P", 0, 2, 1, key="sarima_p")
            seasonal_d = st.number_input("Seasonal D", 0, 2, 1, key="sarima_d")
            seasonal_q = st.number_input("Seasonal Q", 0, 2, 1, key="sarima_q")
            season_length = st.number_input("Season length", 1, 30, 7, key="sarima_length")

            if st.button("Run SARIMA Forecast", key="run_sarima"):
                from src.models.sarima_model import fit_sarima, forecast
                res = fit_sarima(df, order=(1,1,1),
                                 seasonal_order=(seasonal_p, seasonal_d, seasonal_q, season_length))
                yhat = forecast(res, steps)
                future_index = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=steps, freq='D')
                forecast_series = pd.Series(yhat.values, index=future_index)
                st.session_state['forecast_sarima'] = forecast_series
                plot_actual_vs_forecast(df, forecast_series, title="SARIMA Forecast")

        # --- LSTM ---
        elif model_choice == "LSTM":
            window_size = st.slider("Window size (days)", 7, 60, 30, key="lstm_window")
            epochs = st.slider("Training epochs", 10, 100, 20, key="lstm_epochs")
            batch_size = st.slider("Batch size", 8, 64, 16, key="lstm_batch")

            if st.button("Run LSTM Forecast", key="run_lstm"):
                try:
                    from src.models.lstm_model import fit_lstm, forecast_lstm
                    model, scaler = fit_lstm(df['sales'], window_size=window_size, epochs=epochs, batch_size=batch_size)
                    yhat, forecast_index = forecast_lstm(model, scaler, df['sales'], steps=steps, window_size=window_size)
                    forecast_series = pd.Series(yhat, index=forecast_index)
                    st.session_state['forecast_lstm'] = forecast_series
                    plot_actual_vs_forecast(df, forecast_series, title="LSTM Forecast")
                except ValueError as e:
                    st.error(str(e))

# ---------------- Evaluation Tab ----------------
with tab3:
    st.header("Model Evaluation")
    if 'df' in st.session_state:
        df = st.session_state['df']
        if len(df) >= 30:
            y_true = df['sales'].iloc[-30:].values
            results = []
            if 'forecast_arima' in st.session_state:
                y_pred = st.session_state['forecast_arima'].values[:len(y_true)]
                results.append(["ARIMA", round(mape(y_true, y_pred), 2), round(rmse(y_true, y_pred), 2)])
            if 'forecast_sarima' in st.session_state:
                y_pred = st.session_state['forecast_sarima'].values[:len(y_true)]
                results.append(["SARIMA", round(mape(y_true, y_pred), 2), round(rmse(y_true, y_pred), 2)])
            if 'forecast_lstm' in st.session_state:
                y_pred = st.session_state['forecast_lstm'].values[:len(y_true)]
                results.append(["LSTM", round(mape(y_true, y_pred), 2), round(rmse(y_true, y_pred), 2)])
            if results:
                st.write("### Accuracy Comparison")
                st.dataframe(pd.DataFrame(results, columns=["Model", "MAPE", "RMSE"]))
            else:
                st.info("Run forecasts first to see evaluation results.")
        else:
            st.warning("Dataset must have at least 30 records for evaluation.")
    else:
        st.warning("Upload a dataset in the Dataset tab first.")

# ---------------- Visualization Tab ----------------
with tab4:
    st.header("Visualization")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.write("### Sales Over Time")
        st.line_chart(df['sales'])
        st.write("### Seasonal Decomposition")
        from statsmodels.tsa.seasonal import seasonal_decompose
        try:
            result = seasonal_decompose(df['sales'], model="additive", period=7)
            st.pyplot(result.plot())
        except Exception as e:
            st.error(f"Seasonal decomposition failed: {e}")
        st.write("### Forecast Comparison Overlay")
        forecasts = {}
        if 'forecast_arima' in st.session_state:
            forecasts["ARIMA"] = st.session_state['forecast_arima']
        if 'forecast_sarima' in st.session_state:
            forecasts["SARIMA"] = st.session_state['forecast_sarima']
        if 'forecast_lstm' in st.session_state:
            forecasts["LSTM"] = st.session_state['forecast_lstm']
        if forecasts:
            plot_model_overlays(df, forecasts)
        else:
            st.info("Run forecasts first to see overlay comparison.")
    else:
        st.info("Upload a dataset to visualize.")