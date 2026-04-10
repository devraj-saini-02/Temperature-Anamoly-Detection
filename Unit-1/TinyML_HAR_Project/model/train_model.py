import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load processed data
X = np.load("data/X_processed.npy")
y = np.load("data/y_processed.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TinyML-friendly model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(561,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Save model
model.save("model/har_model.h5")
print("Model saved successfully!")