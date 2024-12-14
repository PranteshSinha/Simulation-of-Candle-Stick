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
from collections import defaultdict

# Import pattern detection functions
from Flags import find_flags_pennants_trendline, FlagPattern, fit_trendlines_high_low
from hs_ihs import find_hs_patterns
# Set the matplotlib backend for Streamlit
import matplotlib
matplotlib.use('Agg')


def append_to_csv(entry, file_path):
    # Convert the entry to a DataFrame
    df_entry = pd.DataFrame([entry], columns=["Base Time", "Confirmation Time", "Confirmation Index", "Type", "Trend"])
    
    # Append the entry to the CSV file
    df_entry.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)

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

def plot_flags_pennats(ax, df_window, bull_flags, bear_flags, start_idx, end_idx):
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
                df_window, alines = dict(alines = [[(base_time, flag.base_y), (df_window.index[flag.tip_x - start_idx], flag.tip_y)],
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
                df_window, alines = dict(alines = [[(base_time, flag.base_y), (df_window.index[flag.tip_x - start_idx], flag.tip_y)],
                [(df_window.index[flag.tip_x - start_idx], flag.resist_intercept), (conf_time, flag.resist_intercept + flag.resist_slope * flag.flag_width)],
                [(df_window.index[flag.tip_x - start_idx], flag.support_intercept), (conf_time, flag.support_intercept + flag.support_slope * flag.flag_width)]],
                colors = ['r', 'b', 'b']), type='candle', style='charles', ax=ax
            )

def plot_hs_ihs(ax, df_window, hs_patterns, ihs_patterns, start_idx, end_idx):
    """
    Overlay detected patterns onto the candlestick chart.
    """
    # Get the start and end timestamps of the current window
    start_time = df_window.index[0]
    end_time = df_window.index[-1]
    for pattern in hs_patterns:
        # Map indices to timestamps
        print(f"{start_idx}.....{pattern.start_i}......{pattern.break_i}")
        start_time = df_window.index[pattern.start_i - start_idx] if pattern.start_i >= start_idx and pattern.start_i < end_idx else None
        break_time = df_window.index[pattern.break_i - start_idx] if pattern.break_i >= start_idx and pattern.break_i < end_idx else None
        # print(f"{start_time} {break_time}")
        if start_time and break_time and start_time <= start_time <= end_time and start_time <= break_time <= end_time:
            plt.style.use('dark_background')
            l0 = [(df_window.index[pattern.start_i - start_idx], pattern.neck_start), (df_window.index[pattern.l_shoulder - start_idx], pattern.l_shoulder_p)]
            l1 = [(df_window.index[pattern.l_shoulder - start_idx], pattern.l_shoulder_p), (df_window.index[pattern.l_armpit - start_idx], pattern.l_armpit_p)]
            l2 = [(df_window.index[pattern.l_armpit - start_idx], pattern.l_armpit_p ), (df_window.index[pattern.head - start_idx], pattern.head_p)]
            l3 = [(df_window.index[pattern.head - start_idx], pattern.head_p ), (df_window.index[pattern.r_armpit - start_idx], pattern.r_armpit_p)]
            l4 = [(df_window.index[pattern.r_armpit - start_idx], pattern.r_armpit_p ), (df_window.index[pattern.r_shoulder - start_idx], pattern.r_shoulder_p)]
            l5 = [(df_window.index[pattern.r_shoulder - start_idx], pattern.r_shoulder_p ), (df_window.index[pattern.break_i - start_idx], pattern.neck_end)]
            neck = [(df_window.index[pattern.start_i - start_idx], pattern.neck_start), (df_window.index[pattern.break_i - start_idx], pattern.neck_end)]


            mpf.plot(df_window, alines=dict(alines=[l0, l1, l2, l3, l4, l5, neck ], colors=['w', 'w', 'w', 'w', 'w', 'w', 'r']), type='candle', style='charles', ax=ax)

    for pattern in ihs_patterns:
        # Map indices to timestamps
        print(f"{start_idx}.....{pattern.start_i}......{pattern.break_i}")
        start_time = df_window.index[pattern.start_i - start_idx] if pattern.start_i >= start_idx and pattern.start_i < end_idx else None
        break_time = df_window.index[pattern.break_i - start_idx] if pattern.break_i >= start_idx and pattern.break_i < end_idx else None
        # print(f"{start_time} {break_time}")
        if start_time and break_time and start_time <= start_time <= end_time and start_time <= break_time <= end_time:
            plt.style.use('dark_background')
            l0 = [(df_window.index[pattern.start_i - start_idx], pattern.neck_start), (df_window.index[pattern.l_shoulder - start_idx], pattern.l_shoulder_p)]
            l1 = [(df_window.index[pattern.l_shoulder - start_idx], pattern.l_shoulder_p), (df_window.index[pattern.l_armpit - start_idx], pattern.l_armpit_p)]
            l2 = [(df_window.index[pattern.l_armpit - start_idx], pattern.l_armpit_p ), (df_window.index[pattern.head - start_idx], pattern.head_p)]
            l3 = [(df_window.index[pattern.head - start_idx], pattern.head_p ), (df_window.index[pattern.r_armpit - start_idx], pattern.r_armpit_p)]
            l4 = [(df_window.index[pattern.r_armpit - start_idx], pattern.r_armpit_p ), (df_window.index[pattern.r_shoulder - start_idx], pattern.r_shoulder_p)]
            l5 = [(df_window.index[pattern.r_shoulder - start_idx], pattern.r_shoulder_p ), (df_window.index[pattern.break_i - start_idx], pattern.neck_end)]
            neck = [(df_window.index[pattern.start_i - start_idx], pattern.neck_start), (df_window.index[pattern.break_i - start_idx], pattern.neck_end)]


            mpf.plot(df_window, alines=dict(alines=[l0, l1, l2, l3, l4, l5, neck ], colors=['w', 'w', 'w', 'w', 'w', 'w', 'r']), type='candle', style='charles', ax=ax)

# Function to simulate the dynamic candlestick chart with flags and pennants
def simulate_candlestick_chart_with_flags(df, bull_flags, bear_flags, patterns, window_size=30, interval=0.5):
    fig, (ax_main, ax_volume) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [4, 1]},
        figsize=(10, 8)
    )
    fig.tight_layout(pad=3)

    plt.ion()
    plot_placeholder = st.empty()

    bull_flag_queue = []
    bear_flag_queue = []
    csv = "flag_results.csv"
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

        if patterns:
            plot_flags_pennats(ax_main, df_window, bull_flags, bear_flags, start_idx, end_idx)
        else:
            plot_flags_pennats(ax_main, df_window, bull_flags, bear_flags, start_idx, end_idx)
            start_time = df_window.index[0]
            end_time = df_window.index[-1]
            for flag in bull_flags:
                base_time = df_window.index[flag.base_x - start_idx] if flag.base_x >= start_idx and flag.base_x < end_idx else None
                conf_time = df_window.index[flag.conf_x - start_idx] if flag.conf_x >= start_idx and flag.conf_x < end_idx else None
                if base_time and conf_time and start_time <= base_time <= end_time and start_time <= conf_time <= end_time and not flag.is_processed:
                    bull_flag_queue.append([base_time, conf_time, flag.conf_x, "Bull"])
                    flag.is_processed = True

                    
            for flag in bear_flags:
                base_time = df_window.index[flag.base_x - start_idx] if flag.base_x >= start_idx and flag.base_x < end_idx else None
                conf_time = df_window.index[flag.conf_x - start_idx] if flag.conf_x >= start_idx and flag.conf_x < end_idx else None
                if base_time and conf_time and start_time <= base_time <= end_time and start_time <= conf_time <= end_time and not flag.is_processed:
                    bear_flag_queue.append([base_time, conf_time, flag.conf_x, "Bear"])
                    flag.is_processed = True
            
            
            while len(bull_flag_queue) > 0 and bull_flag_queue[0][2] + 20 == end_idx:
                price_now = df['Close'].iloc[end_idx - 1]
                price_then = df['Close'].iloc[end_idx - 21]
                # print(f"{price_now} {price_then}")
                if price_now < price_then:
                    isflag = "bearish"
                else:
                    isflag = "bullish"
                bull_flag_queue[0].append(isflag)
                append_to_csv(bull_flag_queue[0], csv)
                bull_flag_queue.pop(0)
            
            
            while len(bear_flag_queue) > 0 and bear_flag_queue[0][2] + 20 == end_idx:
                # print(f"{bear_flag_queue[0][2]} {end_idx}")
                price_now = df['Close'].iloc[end_idx - 1]
                price_then = df['Close'].iloc[end_idx - 21]
                # print(f"{price_now} {price_then}")
                if price_now < price_then:
                    isflag = "bearish"
                else:
                    isflag = "bullish"
                bear_flag_queue[0].append(isflag)
                append_to_csv(bear_flag_queue[0], csv)
                bear_flag_queue.pop(0)
            
            

        plot_placeholder.pyplot(fig)
        time.sleep(interval)

    plt.ioff()
    plt.close(fig)

def simulate_candlestick_chart_with_hs(df, hs_patterns, ihs_patterns, patterns, window_size = 50, interval = 0.5):
    fig, (ax_main, ax_volume) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={'height_ratios': [4, 1]},
        figsize=(10, 8)
    )
    fig.tight_layout(pad=3)

    plt.ion()
    plot_placeholder = st.empty()

    csv = "hs_results.csv"
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

        plot_hs_ihs(ax_main, df_window, hs_patterns, ihs_patterns, start_idx, end_idx)

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
    start="2024-04-01",
    end="2024-12-14"
)

st.write(f'You selected: {option} timeframe')

st.title("Pattern Detection")

# Dropdown for pattern selection
pattern_option = st.selectbox(
    "Select a Pattern",
    ["Flag", "Pennant", "Head and Shoulders"]
)

# Display selected pattern
st.write(f"You selected: **{pattern_option}**")

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
        st.warning("Not enough data after the selected starting date to display a 100-candle window.")
    else:
        # Detect flags and pennants
        data_array = df_filtered['Close'].to_numpy()
        bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(data_array, 10)
        if pattern_option == 'Flag':
            # Simulate with detected patterns
            if freq_map[option] == '5m' or freq_map[option] == '15m':
                simulate_candlestick_chart_with_flags(df_filtered, bull_flags, bear_flags, True, window_size=100, interval=0)
            else:
                simulate_candlestick_chart_with_flags(df_filtered, bull_flags, bear_flags, False, window_size=100, interval=0)
        
        elif pattern_option == 'Pennant':
            if freq_map[option] == '5m' or freq_map[option] == '15m':
                simulate_candlestick_chart_with_flags(df_filtered, bull_pennants, bear_pennants, True, window_size=100, interval=0)
            else:
                simulate_candlestick_chart_with_flags(df_filtered, bull_pennants, bear_pennants, False, window_size=100, interval=0)
        
        else:
            hs_patterns, ihs_patterns = find_hs_patterns(data_array, 4, early_find=True)
            print(hs_patterns)
            print(ihs_patterns)
            if freq_map[option] == '5m' or freq_map[option] == '15m':
                simulate_candlestick_chart_with_hs(df_filtered, hs_patterns, ihs_patterns, True, window_size=100, interval=0)
            else:
                simulate_candlestick_chart_with_hs(df_filtered, hs_patterns, ihs_patterns, False, window_size=100, interval=0)
