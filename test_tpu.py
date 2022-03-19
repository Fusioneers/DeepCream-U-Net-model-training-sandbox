# This script is equivalent to test.py except that it uses a tensorflow lite model

# -- Imports --

import os
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from numpy import asarray
from pycoral.utils import edgetpu


# -- Load the training and test data --

# Define variables
image_directory = 'data/images/'
mask_directory = 'data/masks/'

image_extension = 'jpg'
mask_extension = 'tiff'

WIDTH = 256
HEIGHT = 192
image_dataset = []
mask_dataset = []

print("Loading images ...")

# Load images and masks
masks = sorted(os.listdir(mask_directory))
for i, file_name in enumerate(masks):
    if file_name.endswith(mask_extension):
        image = Image.open(image_directory + file_name.replace(mask_extension, image_extension))
        mask = Image.open(mask_directory + file_name)

        # Resize the image
        scaled_image = ImageOps.fit(image, (WIDTH, HEIGHT), Image.ANTIALIAS)
        # Convert it back to an array
        scaled_image_array = asarray(scaled_image)
        scaled_image_array = scaled_image_array.astype('float32')
        # Scale the pixel values to lie between 0 and 1
        scaled_image_array /= 255.0

        # Resize the mask
        scaled_mask = ImageOps.fit(mask, (WIDTH, HEIGHT), Image.ANTIALIAS)
        # Convert it back to an array
        scaled_mask_array = asarray(scaled_mask)
        scaled_mask_array = scaled_mask_array.astype('float32')
        # Scale the pixel values to lie between 0 and 1
        scaled_mask_array /= 255.0

        # Show image and mask (for debug purposes)
        # plt.figure(figsize=(12, 8))
        # plt.subplot(121)
        # plt.title('scaled_image_array')
        # plt.imshow(scaled_image_array)
        # plt.subplot(122)
        # plt.title('scaled_mask_array')
        # plt.imshow(scaled_mask_array)
        # plt.show()

        image_dataset.append(np.array(scaled_image_array))
        mask_dataset.append(np.array(scaled_mask_array))

X_test, y_test = image_dataset, mask_dataset

print("Done!")

# -- Load model and predict --

print("Loading model ...")

# Load TFLite model and allocate tensors.
interpreter = edgetpu.make_interpreter("models/tflite/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Done!")
print("Predicting ...")

# Iterate through the test data, make a prediction for each one and immediately show it
for i in range(0, len(X_test)):
    # Test model on input data.
    interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Display the image with its prediction
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
