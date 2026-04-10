import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# --- 1. SET PATHS ---
# Path to your trained .keras model
MODEL_PATH = '/kaggle/input/models/devrajnsut/temperature-anamoly-detector/tensorflow2/default/1/best_temperature_model.keras'
# Path to the 500-row holdout dataset we segregated
TEST_DATA_PATH = '/kaggle/input/datasets/devrajnsut/temp-anamoly-test/test_data.csv'

# --- 2. HARDCODED LOGIC SETTINGS (From Research Paper) ---
# We use 38.0 as the "Danger" threshold based on your requirement
THRESHOLD_LIMIT = 38.0

# --- 3. HELPER FUNCTIONS FOR CATEGORIZATION ---
def categorize_actual_data(row):
    temps = row[['t1', 't2', 't3', 't4', 't5']].values
    max_t = np.max(temps)
    variance = np.var(temps)
    
    if max_t > 38.0:
        return "High Heat"
    elif variance > 0.9: # Threshold for a 'spike' pattern
        return "Pattern Spike"
    else:
        return "Normal"

def evaluate_performance(label, prediction, data_type):
    if label == 1 and prediction == 1:
        return "Correct (Detected)"
    elif label == 0 and prediction == 0:
        return "Correct (Normal)"
    elif label == 1 and prediction == 0:
        if data_type == "Pattern Spike":
            return "Missed (Pattern Blind)"
        return "Missed (Threshold Failure)"
    else:
        return "False Alarm"

# --- 4. EXECUTION ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
    print("Error: Check if the model and dataset are correctly attached to the Kaggle session.")
else:
    # Load Model and Data
    model = tf.keras.models.load_model(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Prepare Features
    X_test = test_df[['t1', 't2', 't3', 't4', 't5']].values
    y_true = test_df['label'].values
    
    # NN Predictions (Must scale features exactly like training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    nn_probs = model.predict(X_scaled).flatten()
    nn_preds = (nn_probs > 0.5).astype(int)
    
    # Hardcoded Paper Logic Predictions
    max_temps = np.max(X_test, axis=1)
    logic_preds = (max_temps > THRESHOLD_LIMIT).astype(int)
    
    # --- 5. BUILD COMPARISON TABLE ---
    results_df = test_df.copy()
    results_df['max_temp_recorded'] = np.round(max_temps, 2)
    results_df['nn_detected'] = nn_preds
    results_df['logic_detected'] = logic_preds
    
    # Apply Categorization
    results_df['Actual_Data_Type'] = results_df.apply(categorize_actual_data, axis=1)
    
    results_df['NN_Performance'] = results_df.apply(
        lambda r: evaluate_performance(r['label'], r['nn_detected'], r['Actual_Data_Type']), axis=1)
    
    results_df['Paper_Logic_Performance'] = results_df.apply(
        lambda r: evaluate_performance(r['label'], r['logic_detected'], r['Actual_Data_Type']), axis=1)
    
    # Save the Final CSV
    output_filename = 'model_vs_logic_comparison.csv'
    results_df.to_csv(output_filename, index=False)
    
    print(f"Success! Final comparison saved as {output_filename}")
