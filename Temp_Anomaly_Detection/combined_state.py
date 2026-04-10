import pandas as pd

def apply_combined_state(df):
    df = df.copy()

    # 5-state priority machine
    # State 0: No Alert
    # State 1: Spike Alert (pre-threshold early warning) ← YOUR NOVELTY
    # State 2: Level 1 (LED)
    # State 3: Level 2 (LED + Buzzer)
    # State 4: Level 3 (LED + Buzzer + SMS)

    def get_state(row):
        if row['threshold_level'] == 3:
            return 4
        elif row['threshold_level'] == 2:
            return 3
        elif row['threshold_level'] == 1:
            return 2
        elif row['instant_spike']:
            return 1   # fires BEFORE threshold is crossed
        else:
            return 0

    state_labels = {
        0: 'No Alert',
        1: 'Spike Alert (Pre-Threshold)',
        2: 'LED Alert',
        3: 'LED + Buzzer Alert',
        4: 'LED + Buzzer + SMS Alert'
    }

    df['combined_state'] = df.apply(get_state, axis=1)
    df['combined_label'] = df['combined_state'].map(state_labels)
    df['detected_combined'] = df['combined_state'] >= 1

    return df

if __name__ == '__main__':
    from data_loader import load_data
    from baseline_detector import apply_threshold_detector
    from spike_detector import apply_spike_detector
    df = load_data('data/ambient_temperature_system_failure.csv', 'data/combined_windows.json')
    df = apply_threshold_detector(df)
    df = apply_spike_detector(df)
    df = apply_combined_state(df)
    print(df['combined_label'].value_counts())