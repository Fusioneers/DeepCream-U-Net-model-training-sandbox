import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray

# -- Load the test data --

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

# Load the model (and load weights)
model = tf.keras.models.load_model('models/keras')

print("Done!")

# Format variables
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print("Predicting ...")

# Predict
y_pred = model.predict(X_test)

print(y_pred.shape)

# Display the results
for i in range(0, len(y_pred)):
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('X_test example')
    plt.imshow(X_test[i])
    plt.subplot(132)
    plt.title('y_pred example')
    plt.imshow(y_pred[i] > 0.5)
    plt.subplot(133)
    plt.title('y_test example')
    plt.imshow(y_test[i])
    plt.show()

print("Done!")
