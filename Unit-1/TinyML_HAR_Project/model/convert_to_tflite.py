import tensorflow as tf

model = tf.keras.models.load_model("model/har_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model/har_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TinyML model created!")
