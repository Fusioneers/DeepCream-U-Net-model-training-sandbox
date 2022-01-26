import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from simple_unet_model import simple_unet_model

# -- Load the test data --

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
    if file_name.endswith('TIF') and i < 20:
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

# Load the model (and load weights)
model = simple_unet_model(HEIGHT, WIDTH, 3)
model.load_weights('models/keras')

# Format variables
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# Predict
y_pred = model.predict(X_test)

# Display the first 10 results
for i in range(len(y_pred) - 1):
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
