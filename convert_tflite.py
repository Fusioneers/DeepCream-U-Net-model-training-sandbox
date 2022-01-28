import pathlib
import tensorflow as tf

from simple_unet_model import simple_unet_model

WIDTH = 256
HEIGHT = 256

# Load the model (and load weights)
model = simple_unet_model(HEIGHT, WIDTH, 3)
model.load_weights('models/keras')

# Convert the saved model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Print the signatures from the converted model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print(signatures)

tflite_model_files = pathlib.Path('models/tflite/script.tflite')
tflite_model_files.write_bytes(tflite_model)

# tflite_convert --output_file=models/tflite/model2.tflite --keras_model_file=models/keras --input_arrays=input
# --output_arrays=output --experimental_new_converter=False