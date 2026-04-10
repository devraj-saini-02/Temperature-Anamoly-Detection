import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

COLORS = {0: 'green', 1: 'gold', 2: 'orange', 3: 'red'}
T1, T2, T3 = 75.0, 80.0, 84.0

def graph1_raw_profile(df):
    T1 = df.attrs.get('T1', 75.0)
    T2 = df.attrs.get('T2', 80.0)
    T3 = df.attrs.get('T3', 84.0)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['timestamp'], df['temp'], color='steelblue', linewidth=0.8, label='Temperature')

    # Shade anomaly windows
    in_anomaly = False
    start = None
    for i, row in df.iterrows():
        if row['ground_truth'] and not in_anomaly:
            start = row['timestamp']
            in_anomaly = True
        elif not row['ground_truth'] and in_anomaly:
            ax.axvspan(start, row['timestamp'], alpha=0.25, color='red', label='Known Anomaly' if start == df[df['ground_truth']].iloc[0]['timestamp'] else '')
            in_anomaly = False

    ax.set_title('Graph 1 — Raw Temperature Profile with Anomaly Windows', fontsize=13)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/graph1_raw_profile.png', dpi=150)
    plt.close()
    print("Saved: graph1_raw_profile.png")

def graph2_threshold_overlay(df):
    T1 = df.attrs.get('T1', 75.0)
    T2 = df.attrs.get('T2', 80.0)
    T3 = df.attrs.get('T3', 84.0)
    fig, ax = plt.subplots(figsize=(14, 4))

    # Color segments by threshold level
    level_colors = {0: 'green', 1: 'gold', 2: 'orange', 3: 'red'}
    for i in range(len(df) - 1):
        lvl = df['threshold_level'].iloc[i]
        ax.plot(df['timestamp'].iloc[i:i+2],
                df['temp'].iloc[i:i+2],
                color=level_colors[lvl], linewidth=0.9)

    ax.axhline(T1, color='gold',   linestyle='--', linewidth=1, label=f'T1={T1}°C')
    ax.axhline(T2, color='orange', linestyle='--', linewidth=1, label=f'T2={T2}°C')
    ax.axhline(T3, color='red',    linestyle='--', linewidth=1, label=f'T3={T3}°C')

    patches = [mpatches.Patch(color=c, label=l) for c, l in
               [('green','Safe'), ('gold','Level 1'), ('orange','Level 2'), ('red','Level 3')]]
    ax.legend(handles=patches + ax.get_lines()[:3], fontsize=8)
    ax.set_title('Graph 2 — Multi-level Threshold Detection (LM35 Paper Style)', fontsize=13)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig('outputs/graph2_threshold_overlay.png', dpi=150)
    plt.close()
    print("Saved: graph2_threshold_overlay.png")

def graph3_spike_zoom(df):
    # Zoom into failure region
    T1 = df.attrs.get('T1', 75.0)
    T2 = df.attrs.get('T2', 80.0)
    T3 = df.attrs.get('T3', 84.0)
    anomaly_df = df[df['ground_truth']]
    if len(anomaly_df) == 0:
        print("No anomaly window found for zoom.")
        return

    center = anomaly_df['timestamp'].iloc[len(anomaly_df)//2]
    mask = (df['timestamp'] >= center - pd.Timedelta(hours=48)) & \
           (df['timestamp'] <= center + pd.Timedelta(hours=48))
    zoom = df[mask]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom['timestamp'], zoom['temp_smooth'], color='steelblue', linewidth=1.2, label='Smoothed Temp')
    ax.plot(zoom['timestamp'], zoom['temp'], color='lightblue', linewidth=0.5, alpha=0.5, label='Raw Temp')

    # Spike markers
    spikes = zoom[zoom['instant_spike']]
    ax.scatter(spikes['timestamp'], spikes['temp'], color='red', s=40, zorder=5,
               label=f'Spike Detected ({len(spikes)})')

    # Threshold breach markers
    thresh = zoom[zoom['detected_threshold']]
    ax.scatter(thresh['timestamp'], thresh['temp'], color='orange', s=20, marker='^',
               zorder=4, label=f'Threshold Breach ({len(thresh)})')

    ax.axhline(T1, color='gold', linestyle='--', linewidth=1, label=f'T1={T1}°C')
    ax.set_title('Graph 3 — Zoomed: Spike Detection vs Threshold in Failure Region', fontsize=13)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/graph3_spike_zoom.png', dpi=150)
    plt.close()
    print("Saved: graph3_spike_zoom.png")

def graph4_alert_state_timeline(df):
    T1 = df.attrs.get('T1', 75.0)
    T2 = df.attrs.get('T2', 80.0)
    T3 = df.attrs.get('T3', 84.0)
    state_colors = {0: 'green', 1: 'blue', 2: 'gold', 3: 'orange', 4: 'red'}
    state_labels = {
        0: 'No Alert',
        1: 'Spike (Pre-Threshold)',
        2: 'LED Alert',
        3: 'LED+Buzzer',
        4: 'LED+Buzzer+SMS'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top: temperature
    ax1.plot(df['timestamp'], df['temp'], color='steelblue', linewidth=0.7)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Graph 4 — Combined Alert State Machine Timeline', fontsize=13)

    # Bottom: state color band
    for i in range(len(df) - 1):
        state = df['combined_state'].iloc[i]
        ax2.axvspan(df['timestamp'].iloc[i], df['timestamp'].iloc[i+1],
                    alpha=0.7, color=state_colors[state])

    patches = [mpatches.Patch(color=state_colors[s], label=state_labels[s]) for s in range(5)]
    ax2.legend(handles=patches, loc='upper left', fontsize=8, ncol=5)
    ax2.set_ylabel('Alert State')
    ax2.set_yticks([])
    ax2.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('outputs/graph4_alert_timeline.png', dpi=150)
    plt.close()
    print("Saved: graph4_alert_timeline.png")

if __name__ == '__main__':
    from data_loader import load_data
    from baseline_detector import apply_threshold_detector
    from spike_detector import apply_spike_detector
    from combined_state import apply_combined_state
    from evaluation import evaluate

    df = load_data('data/ambient_temperature_system_failure.csv', 'data/combined_windows.json')
    df = apply_threshold_detector(df)
    df = apply_spike_detector(df)
    df = apply_combined_state(df)
    _, _, df = evaluate(df)

    graph1_raw_profile(df)
    graph2_threshold_overlay(df)
    graph3_spike_zoom(df)
    graph4_alert_state_timeline(df)