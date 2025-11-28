import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

@st.cache_data
def clean_data(df):
    return df.dropna()