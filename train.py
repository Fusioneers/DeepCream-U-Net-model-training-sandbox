# This script trains an AI on the test data and output the keras model into the models/keras folder

# -- Imports --

import os
import numpy as np
import tensorflow
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from numpy import asarray
from sklearn.model_selection import train_test_split
from unet_model import unet_model

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

# Split training and test set
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.1)

# -- Train --

# Load the model (and load weights)
model = unet_model(HEIGHT, WIDTH, 3)  # instantiate a new model
# model = tensorflow.keras.models.load_model('models/keras')  # Load an existing model and continue its training

# Format training and test data
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Train the model
history = model.fit(X_train, y_train, verbose=1, epochs=15, validation_data=(X_train, y_train))

# Save the model
model.save('models/keras')

# -- Evaluate the model --

# Print out accuracy
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Calculate IoU and print it out (might not be representative for small X_test)
y_pred = model.predict(X_test)

y_pred = y_pred.reshape(y_pred.shape[0], HEIGHT, WIDTH)

y_pred_threshold = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_threshold)
union = np.logical_or(y_test, y_pred_threshold)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)
