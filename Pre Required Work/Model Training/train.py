import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 1. Load the Full Dataset
data_path = '/kaggle/input/datasets/devrajnsut/temperature-anamoly-detection/temperature_anamoly_detection.csv'

if not os.path.exists(data_path):
    print("Dataset not found at the specified path.")
else:
    full_df = pd.read_csv(data_path)
    
    # 2. Carve out 500 rows for "True Testing" (Holdout Set)
    # We use a fixed random_state so this set is reproducible
    test_holdout_df = full_df.sample(n=500, random_state=42)
    train_pool_df = full_df.drop(test_holdout_df.index)
    
    # Save the 500 rows to a new CSV for later
    test_holdout_df.to_csv('test_data.csv', index=False)
    print(f"Saved 500 rows to 'holdout_test_data.csv'.")
    
    # 3. Prepare Training/Validation Data (from the remaining 5500)
    X = train_pool_df.drop(columns=['label']).values
    y = train_pool_df['label'].values
    
    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the 5500 rows: 80% Train (4400), 20% Val (1100)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 4. Build the 3-Layer Model
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # 5. Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 6. Callbacks to save the best model during training
    checkpoint_cb = callbacks.ModelCheckpoint(
        "best_temperature_model.keras", 
        save_best_only=True, 
        monitor='val_loss'
    )

    # 7. Train
    print("\nStarting training on 5500 rows...")
    model.fit(
        X_train, y_train, 
        epochs=60, 
        batch_size=32, 
        validation_data=(X_val, y_val), 
        callbacks=[checkpoint_cb],
        verbose=1
    )

    print("\nTraining Complete. 'best_temperature_model.keras' is ready.")