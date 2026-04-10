import pandas as pd
import numpy as np

WINDOW_SIZE  = 5
SPIKE_DELTA  = 1.5    # lowered — dataset has subtle changes

def apply_spike_detector(df):
    df = df.copy()

    # Smoothed temp
    df['temp_smooth'] = df['temp'].rolling(window=3, center=True, min_periods=1).mean()

    # Delta = current - min of previous window
    rolling_min = df['temp_smooth'].rolling(window=WINDOW_SIZE, min_periods=1).min().shift(1)
    df['delta_window'] = (df['temp_smooth'] - rolling_min).fillna(0)

    # Spike flag
    df['instant_spike'] = df['delta_window'] >= SPIKE_DELTA

    # Also flag sudden DROPS (anomalies here include temp drops during failure)
    rolling_max = df['temp_smooth'].rolling(window=WINDOW_SIZE, min_periods=1).max().shift(1)
    df['delta_drop'] = (rolling_max - df['temp_smooth']).fillna(0)
    df['instant_drop'] = df['delta_drop'] >= SPIKE_DELTA

    # Combined spike flag = rise OR drop
    df['instant_spike'] = df['instant_spike'] | df['instant_drop']

    print(f"Spike/drop events: {df['instant_spike'].sum()}")
    return df