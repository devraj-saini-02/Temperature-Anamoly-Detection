import pandas as pd
import numpy as np

def apply_threshold_detector(df):
    df = df.copy()

    # Use statistical thresholds based on actual data distribution
    mean_temp = df['temp'].mean()
    std_temp  = df['temp'].std()

    T1 = mean_temp + 0.5 * std_temp   # mild deviation
    T2 = mean_temp + 1.0 * std_temp   # moderate
    T3 = mean_temp + 1.5 * std_temp   # critical

    print(f"Thresholds → T1: {T1:.2f}, T2: {T2:.2f}, T3: {T3:.2f} °C")
    print(f"Data mean: {mean_temp:.2f}, std: {std_temp:.2f}")

    df['threshold_level'] = 0
    df.loc[df['temp'] >= T1, 'threshold_level'] = 1
    df.loc[df['temp'] >= T2, 'threshold_level'] = 2
    df.loc[df['temp'] >= T3, 'threshold_level'] = 3

    level_labels  = {0:'Safe', 1:'Level 1 Alert', 2:'Level 2 Alert', 3:'Level 3 Alert'}
    level_actions = {0:'No Alert', 1:'LED Alert', 2:'LED + Buzzer Alert', 3:'LED + Buzzer + SMS Alert'}

    df['alert_label']  = df['threshold_level'].map(level_labels)
    df['alert_action'] = df['threshold_level'].map(level_actions)
    df['detected_threshold'] = df['threshold_level'] >= 1

    # Store thresholds for use in graphs
    df.attrs['T1'] = T1
    df.attrs['T2'] = T2
    df.attrs['T3'] = T3

    return df