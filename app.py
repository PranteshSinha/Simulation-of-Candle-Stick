import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import StringIO
import time
from datetime import datetime

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

# Function to simulate the dynamic candlestick chart (window size of 30)
def simulate_candlestick_chart(df, window_size=30, interval=2):
    """
    Simulate a dynamic candlestick chart with a fixed moving window of size 30.
    """
    # Create a figure with two subplots (main chart and volume)
    fig, (ax_main, ax_volume) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [4, 1]},  # Main chart taller than volume
        figsize=(10, 8)
    )
    fig.tight_layout(pad=3)  # Adjust spacing for better layout

    # Set up the plot
    plt.ion()  # Turn on interactive mode

    # Create a Streamlit empty container for the plot
    plot_placeholder = st.empty()

    # Loop to update the plot with a moving window of 30
    for start_idx in range(len(df) - window_size + 1):
        end_idx = start_idx + window_size
        df_window = df.iloc[start_idx:end_idx]  # Get the moving window

        # Clear both axes
        ax_main.clear()
        ax_volume.clear()

        # Plot the candlestick chart with a custom x-axis date format (dd-mm-yyyy)
        mpf.plot(
            df_window,
            type='candle',
            style='charles',
            ax=ax_main,
            volume=ax_volume,  # Assign the volume subplot
        )

        # Update the plot in Streamlit
        plot_placeholder.pyplot(fig)

        # Wait for the specified interval (in seconds)
        time.sleep(interval)  # 10-second delay

    # Adjust x-axis labels format (dd-mm-yyyy)
    ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m-%Y'))

    plt.ioff()  # Turn off interactive mode
    plt.close(fig)  # Close the figure after the simulation

# Streamlit UI
st.title("Bitcoin Candlestick Chart")

# File uploader for user to upload their dataset
file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])

if file is not None:
    # Option to choose the time frame
    option = st.selectbox(
        'Select Time Frame',
        ('Daily', 'Weekly', 'Monthly')
    )
    st.write(f'You selected: {option} timeframe')

    # Process the data based on selected timeframe
    df_resampled = process_data(file, freq=option[0])

    # Check if the data is valid
    if df_resampled is None:
        st.error("Failed to process the data. Please check the file format or contents.")
    else:
        # Get the earliest date in the dataset
        min_date = df_resampled.index.min().date()

        # Add a date picker for the starting date
        start_date = st.date_input(
            "Select Starting Date",
            value=min_date,  # Default to the earliest date
            min_value=min_date,  # Earliest date allowed
            max_value=df_resampled.index.max().date()  # Latest date allowed
        )

        # Filter the data based on the selected starting date
        df_filtered = df_resampled[df_resampled.index >= pd.Timestamp(start_date)]

        # Check if there's enough data for the selected starting date
        if len(df_filtered) < 30:
            st.warning("Not enough data after the selected starting date to display a 30-day window.")
        else:
            # Simulate and display the dynamic candlestick chart with 30 window size and 10-second delay
            simulate_candlestick_chart(df_filtered, window_size=30, interval=0.5)
