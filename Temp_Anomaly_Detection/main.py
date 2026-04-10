from data_loader import load_data
from baseline_detector import apply_threshold_detector
from spike_detector import apply_spike_detector
from combined_state import apply_combined_state
from evaluation import evaluate
from graphs import (graph1_raw_profile, graph2_threshold_overlay,
                    graph3_spike_zoom, graph4_alert_state_timeline)

print("=== HEAT ANOMALY DETECTION PIPELINE ===\n")

df = load_data('data/ambient_temperature_system_failure.csv', 'data/combined_windows.json')
print(f"Data loaded: {len(df)} rows, {df['ground_truth'].sum()} anomaly points\n")

df = apply_threshold_detector(df)
df = apply_spike_detector(df)
df = apply_combined_state(df)

print(f"Spike events detected: {df['instant_spike'].sum()}")
print(f"Combined detections:   {df['detected_combined'].sum()}\n")

table1, table2, df = evaluate(df)

graph1_raw_profile(df)
graph2_threshold_overlay(df)
graph3_spike_zoom(df)
graph4_alert_state_timeline(df)

print("\n=== DONE. Check outputs/ folder for graphs. ===")