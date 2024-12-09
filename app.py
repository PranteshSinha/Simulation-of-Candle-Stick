import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import StringIO
import time
from datetime import datetime
import numpy as np

# Import pattern detection functions
from Flags import find_flags_pennants_trendline, FlagPattern

# Set the matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')

# Function to process the data
def process_data(file, freq='D'):
    """
    Process the raw Bitcoin historical data for plotting.
    """
    # Load the data
    df = pd.read_csv(file)

    # Convert 'Date' to datetime format (MM/DD/YYYY) and handle errors gracefully
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

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
    ohlc_cols = ['Open', 'High', 'Low', 'Price']
    for col in ohlc_cols:
        df[col] = df[col].str.replace(',', '').astype(float)

    # Rename 'Price' to 'Close' for clarity
    df.rename(columns={'Price': 'Close'}, inplace=True)

    # Clean and convert 'Vol.' (Volume) column and rename it to 'Volume'
    df['Vol.'] = df['Vol.'].str.replace(',', '').str.replace('K', 'e3').astype(float)
    df.rename(columns={'Vol.': 'Volume'}, inplace=True)

    # Resample the data based on the selected frequency
    if freq == 'D':  # Daily
        df_resampled = df
    elif freq == 'W':  # Weekly
        df_resampled = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif freq == 'M':  # Monthly
        df_resampled = df.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    else:
        raise ValueError("Invalid frequency. Use 'D', 'W', or 'M'.")

    # Check if the data is empty after resampling
    if df_resampled.empty:
        st.error("No valid data available after resampling.")
        return None  # Return None if no resampled data

    return df_resampled

# Function to plot detected patterns
def plot_patterns(ax, df_window, bull_flags, bear_flags, start_idx, end_idx):
    """
    Overlay detected patterns onto the candlestick chart.
    """
    # Get the start and end timestamps of the current window
    start_time = df_window.index[0]
    end_time = df_window.index[-1]

    for flag in bull_flags:
        # Map indices to timestamps
        print(f"{start_idx}.....{flag.base_x}......{flag.conf_x}")
        base_time = df_window.index[flag.base_x - start_idx] if flag.base_x >= start_idx and flag.base_x < end_idx else None
        conf_time = df_window.index[flag.conf_x - start_idx] if flag.conf_x >= start_idx and flag.conf_x < end_idx else None
        # print(f"{base_time} {conf_time}")
        if base_time and conf_time and start_time <= base_time <= end_time and start_time <= conf_time <= end_time:
            plt.style.use('dark_background')
            mpf.plot(
                df_window, alines = dict(alines = [[(base_time, flag.base_y - start_idx), (df_window.index[flag.tip_x - start_idx], flag.tip_y - start_idx)],
                [(df_window.index[flag.tip_x - start_idx], flag.resist_intercept), (conf_time, flag.resist_intercept + flag.resist_slope * flag.flag_width)],
                [(df_window.index[flag.tip_x - start_idx], flag.support_intercept), (conf_time, flag.support_intercept + flag.support_slope * flag.flag_width)]],
                colors = ['g', 'b', 'b']), type='candle', style='charles', ax=ax
            )

    for flag in bear_flags:
        # Map indices to timestamps
        print(f"{start_idx}.....{flag.base_x}......{flag.conf_x}")
        base_time = df_window.index[flag.base_x - start_idx] if flag.base_x >= start_idx and flag.base_x < end_idx else None
        conf_time = df_window.index[flag.conf_x - start_idx] if flag.conf_x >= start_idx and flag.conf_x < end_idx else None
        # print(f"{base_time} {conf_time}")
        if base_time and conf_time and start_time <= base_time <= end_time and start_time <= conf_time <= end_time:
            plt.style.use('dark_background')
            mpf.plot(
                df_window, alines = dict(alines = [[(base_time, flag.base_y - start_idx), (df_window.index[flag.tip_x - start_idx], flag.tip_y - start_idx)],
                [(df_window.index[flag.tip_x - start_idx], flag.resist_intercept), (conf_time, flag.resist_intercept + flag.resist_slope * flag.flag_width)],
                [(df_window.index[flag.tip_x - start_idx], flag.support_intercept), (conf_time, flag.support_intercept + flag.support_slope * flag.flag_width)]],
                colors = ['r', 'b', 'b']), type='candle', style='charles', ax=ax
            )


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
        plt.style.use('dark_background')
        mpf.plot(
            df_window,
            type='candle',
            style='charles',
            ax=ax_main,
            volume=ax_volume,
        )

        plot_patterns(ax_main, df_window, bull_flags, bear_flags, start_idx, end_idx)

        plot_placeholder.pyplot(fig)
        time.sleep(interval)

    plt.ioff()
    plt.close(fig)

# Streamlit UI
st.title("Bitcoin Candlestick Chart with Flags and Pennants")

file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])

if file is not None:
    option = st.selectbox('Select Time Frame', ('Daily', 'Weekly', 'Monthly'))
    st.write(f'You selected: {option} timeframe')

    df_resampled = process_data(file, freq=option[0])

    if df_resampled is not None:
        min_date = df_resampled.index.min().date()

        start_date = st.date_input(
            "Select Starting Date",
            value=min_date,
            min_value=min_date,
            max_value=df_resampled.index.max().date()
        )

        df_filtered = df_resampled[df_resampled.index >= pd.Timestamp(start_date)]
        
        if len(df_filtered) < 100:
            st.warning("Not enough data after the selected starting date to display a 30-day window.")
        else:
            # Detect flags and pennants
            data_array = df_filtered['Close'].to_numpy()
            bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(data_array, 10)

            # Simulate with detected patterns
            simulate_candlestick_chart_with_patterns(df_filtered, bull_flags, bear_flags, window_size=200, interval=0)
