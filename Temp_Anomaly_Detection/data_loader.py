import pandas as pd
import numpy as np
import json

def load_data(csv_path, labels_path):
    # Load temperature data
    df = pd.read_csv(csv_path)
    df.columns = ['timestamp', 'temp']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Drop duplicates
    df = df.drop_duplicates(subset='timestamp')

    # Interpolate missing values
    df = df.set_index('timestamp').resample('5min').interpolate(method='time').reset_index()

    # Load NAB anomaly windows
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    key = [k for k in labels.keys() if 'ambient_temperature' in k][0]
    windows = labels[key]

    # Tag ground truth
    df['ground_truth'] = False
    for window in windows:
        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'ground_truth'] = True

    # Smoothed temperature (3-point moving average)
    df['temp_smooth'] = df['temp'].rolling(window=3, center=True).mean().fillna(df['temp'])

    return df

if __name__ == '__main__':
    df = load_data('data/ambient_temperature_system_failure.csv', 'data/combined_windows.json')
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Anomaly points: {df['ground_truth'].sum()}")
    print(f"Temp range: {df['temp'].min():.2f} - {df['temp'].max():.2f} °C")