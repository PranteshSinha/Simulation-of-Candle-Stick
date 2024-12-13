import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import StringIO
import time
from datetime import datetime
import numpy as np
# pip install pricehub
from pricehub import get_ohlc

# Import pattern detection functions
from Flags import find_flags_pennants_trendline, FlagPattern

# Set the matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')

# Function to process the data
def process_data(df, freq='1d'):
    """
    Process the raw Bitcoin historical data for plotting.
    """
    # Rename columns to match expected format
    df.rename(columns={
        'Close time': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)

    # Convert 'Date' to datetime format and handle errors gracefully
    df['Date'] = pd.to_datetime(df['Date'], unit='ms', errors='coerce')

    # Drop rows where the date conversion failed (NaT)
    df.dropna(subset=['Date'], inplace=True)

    # Check if data exists after cleaning
    if df.empty:
        st.error("No valid data available after date parsing.")
        return None  # Return None if no data

    # Set 'Date' as index and sort by Date in ascending order
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Clean and convert OHLC columns to numeric
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean and convert 'Volume' column to numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # No additional resampling needed as data is already fetched with the correct interval
    return df

# Function to simulate the dynamic candlestick chart with flags and pennants
def simulate_candlestick_chart_with_patterns(df, bull_flags, bear_flags, window_size=30, interval=0.5):
    fig, (ax_main, ax_volume) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [4, 1]},
        figsize=(10, 8)
    )
    fig.tight_layout(pad=3)

    plt.ion()
    plot_placeholder = st.empty()

    for start_idx in range(len(df) - window_size + 1):
        end_idx = start_idx + window_size
        df_window = df.iloc[start_idx:end_idx]

        ax_main.clear()
        ax_volume.clear()

        mpf.plot(
            df_window,
            type='candle',
            style='charles',
            ax=ax_main,  # Correct argument for the main chart
            volume=ax_volume,  # Correct argument for the volume subplot
        )

        plot_placeholder.pyplot(fig)
        time.sleep(interval)

    plt.ioff()
    plt.close(fig)

# Streamlit UI
st.title("Bitcoin Candlestick Chart with Flags and Pennants")

option = st.selectbox('Select Time Frame', ('Daily', 'Weekly', '15 Minutes', '5 Minutes', '4 Hours'))
freq_map = {'Daily': '1d', 'Weekly': '1w', '15 Minutes': '15m', '5 Minutes': '5m', '4 Hours': '4h'}

file = get_ohlc(
    broker="binance_spot",
    symbol="BTCUSDT",
    interval=freq_map[option],
    start="2024-10-01",
    end="2024-12-10"
)
st.write(f'You selected: {option} timeframe')
df_processed = process_data(file, freq=freq_map[option])
if df_processed is not None:
    min_date = df_processed.index.min().date()
    start_date = st.date_input(
        "Select Starting Date",
        value=min_date,
        min_value=min_date,
        max_value=df_processed.index.max().date()
    )

    df_filtered = df_processed[df_processed.index >= pd.Timestamp(start_date)]

    if len(df_filtered) < 100:
        st.warning("Not enough data after the selected starting date to display a 30-day window.")
    else:
        # Detect flags and pennants
        data_array = df_filtered['Close'].to_numpy()
        bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(data_array, 10)

        # Simulate with detected patterns
        simulate_candlestick_chart_with_patterns(df_filtered, bull_flags, bear_flags, window_size=100, interval=0.5)
