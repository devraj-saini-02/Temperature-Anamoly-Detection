import pandas as pd
import numpy as np

def sigmoid(x, midpoint=38.0, steepness=12):
    """
    High steepness (12) creates a very slight blur. 
    37.8 is likely 0, 38.0 is 50/50, 38.2 is likely 1.
    """
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def generate_dataset(num_rows=6000):
    np.random.seed(42)
    rows = []
    
    for _ in range(num_rows):
        # We split the data into 5 specific scenarios to force the NN to learn patterns
        scenario = np.random.choice(['normal', 'low_spike', 'high_stable', 'extreme', 'mid_blur'], 
                                    p=[0.35, 0.15, 0.15, 0.15, 0.20])
        
        if scenario == 'normal':
            # 30-33°C: Perfectly stable
            seq = np.random.uniform(30.0, 33.0, 5)
            label = 0
            
        elif scenario == 'low_spike':
            # Pattern Anomaly: 30°C base but jumps to ~32.8 suddenly
            base = np.random.uniform(29.8, 30.2)
            seq = np.array([base, base + 0.1, base + 2.7, base - 0.1, base + 0.2]) 
            seq += np.random.normal(0, 0.05, 5) # add tiny noise
            label = 1
            
        elif scenario == 'high_stable':
            # High but Safe: Stable around 37.0 - 37.6
            seq = np.random.uniform(37.0, 37.6, 5)
            label = 0
            
        elif scenario == 'extreme':
            # Threshold Anomaly: Anything significantly over 38.5 (up to 42)
            seq = np.random.uniform(38.8, 42.0, 5)
            label = 1
            
        else: # 'mid_blur' scenario (35 - 38.5)
            # This is where the NN has to work hardest
            seq = np.random.uniform(35.0, 38.5, 5)
            max_val = np.max(seq)
            variance = np.var(seq)
            
            # 1. Pattern Logic: If the variance is high (unstable), it's an anomaly
            # 2. Threshold Logic: Use Sigmoid for the 38.0 boundary
            p_threshold = sigmoid(max_val)
            is_erratic = 1 if variance > 0.9 else 0
            
            # Combine: Anomaly if it's over the blurred threshold OR if the pattern is erratic
            prob = max(p_threshold, is_erratic)
            label = 1 if np.random.random() < prob else 0

        rows.append(list(np.round(seq, 2)) + [int(label)])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['t1', 't2', 't3', 't4', 't5', 'label'])
    
    # Final Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save
    df.to_csv('temperature_anamoly_detection.csv', index=False)
    print(f"Dataset saved! Total rows: {len(df)}")
    print(f"Anomalies (1): {df['label'].sum()} | Normal (0): {len(df) - df['label'].sum()}")
    return df

if __name__ == "__main__":
    df = generate_dataset(6000)
    
    # Quick Check on the logic
    print("\n--- Logic Verification ---")
    print("Example of a low-temp spike (Should be 1):")
    print(df[(df['t3'] > 32.5) & (df['t1'] < 31)].head(1))
    
    print("\nExample of stable high heat (Should be 0):")
    print(df[(df['t1'] > 37.0) & (df['t1'] < 37.6) & (df['label'] == 0)].head(1))