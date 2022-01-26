import os
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
from pycoral.utils import edgetpu
from sklearn.model_selection import train_test_split

# -- Load the training and test data --

# Define variables
image_directory = 'data/generated/generated_patches/images/'
mask_directory = 'data/generated/generated_patches/masks/'

WIDTH = 128 * 2
HEIGHT = 128 * 2
image_dataset = []
mask_dataset = []

# Load images and masks
images = sorted(os.listdir(image_directory))
for i, file_name in enumerate(images):
    if file_name.endswith('TIF') and i < 50:
        image = Image.open(image_directory + file_name)
        mask = Image.open(mask_directory + file_name)

        image.thumbnail((WIDTH, HEIGHT))
        mask.thumbnail((WIDTH, HEIGHT))

        image_pixels = asarray(image)
        image_pixels = image_pixels.astype('float32')
        image_pixels /= 255.0
        mask_pixels = asarray(mask)
        mask_pixels = mask_pixels.astype('float32')
        mask_pixels /= 255.0

        image_dataset.append(np.array(image_pixels))
        mask_dataset.append(np.array(mask_pixels))

X_test, y_test = image_dataset, mask_dataset

# -- Predict --

# Load TFLite model and allocate tensors.
interpreter = edgetpu.make_interpreter("models/tflite/script.tflite")
# interpreter = tf.lite.Interpreter(
#     model_path="models/tflite/script.tflite", model_content=None, experimental_delegates=None,
#     num_threads=None,
#     experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO,
#     experimental_preserve_all_tensors=False
# )
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

# Test model on input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], [X_test[10]])

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
print(interpreter.get_tensor(97))
output_data = interpreter.get_tensor(97)

# Display the first 10 results
plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.title('X_test example')
plt.imshow(X_test[10])
plt.subplot(132)
plt.title('y_pred example')
plt.imshow(output_data[0] > 0.5)
plt.subplot(133)
plt.title('y_test example')
plt.imshow(y_test[10])
plt.show()
