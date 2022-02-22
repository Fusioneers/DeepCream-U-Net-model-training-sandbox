import os
import random

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
from pycoral.utils import edgetpu

# -- Load the training and test data --

# Define variables
image_directory = 'data/hand_annotated_2/generated_patches/images/'
mask_directory = 'data/hand_annotated_2/generated_patches/masks/'

WIDTH = 256
HEIGHT = 192
image_dataset = []
mask_dataset = []

print("Loading images ...")

# Load images and masks
masks = sorted(os.listdir(mask_directory))
for i, file_name in enumerate(masks):
    if file_name.endswith('tiff'):
        image = Image.open(image_directory + file_name.replace('tiff', 'jpg'))
        mask = Image.open(mask_directory + file_name)

        image.thumbnail((WIDTH, HEIGHT))
        mask.thumbnail((WIDTH, HEIGHT))

        image_pixels = asarray(image)
        image_pixels = image_pixels.astype('float32')
        image_pixels /= 255.0
        mask_pixels = asarray(mask)
        mask_pixels = mask_pixels.astype('float32')
        mask_pixels /= 255.0

        # Show image and mask (for debug purposes)
        # plt.figure(figsize=(12, 8))
        # plt.subplot(121)
        # plt.title('image_pixels')
        # plt.imshow(image_pixels)
        # plt.subplot(122)
        # plt.title('mask_pixels')
        # plt.imshow(mask_pixels)
        # plt.show()

        image_dataset.append(np.array(image_pixels))
        mask_dataset.append(np.array(mask_pixels))

X_test, y_test = image_dataset, mask_dataset

print("Done!")

# -- Load model and predict --

print("Loading model ...")

# Load TFLite model and allocate tensors.
interpreter = edgetpu.make_interpreter("models/tflite/model.tflite")
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

print("Done!")
print("Predicting ...")

for i in range(0, len(X_test)):
    print(np.array([X_test[i]]).shape)
    # Test model on input data.
    interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display the first 10 results
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('X_test example')
    plt.imshow(X_test[i])
    plt.subplot(132)
    plt.title('y_pred example')
    plt.imshow(output_data[0] > 0.5)
    plt.subplot(133)
    plt.title('y_test example')
    plt.imshow(y_test[i])
    plt.show()

print("Done!")
