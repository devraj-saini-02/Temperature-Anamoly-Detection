import tensorflow as tf
import numpy as np

# Load TinyML model
interpreter = tf.lite.Interpreter(model_path="model/har_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Simulated sensor input (561 features)
sample_input = np.expand_dims(np.load("data/X_processed.npy")[0], axis=0).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
prediction = np.argmax(output)

activities = ['SITTING', 'STANDING', 'WALKING']
print("Predicted Activity:", activities[prediction])