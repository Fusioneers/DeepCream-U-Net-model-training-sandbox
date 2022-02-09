import pathlib
import tensorflow as tf

WIDTH = 256
HEIGHT = 256

print("Loading model ...")

# Load the model (and load weights)
model = tf.keras.models.load_model('models/keras')

print("Done!")
print("Converting ...")

# Convert the saved model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Print the signatures from the converted model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()

print("Done!")
print("Saving ...")

# Writes the file
tflite_model_files = pathlib.Path('models/tflite/model.tflite')
tflite_model_files.write_bytes(tflite_model)

print("Done!")
