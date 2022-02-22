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
image_directory = 'data/auto_annotated/generated_patches/images/'
mask_directory = 'data/auto_annotated/generated_patches/masks/'

WIDTH = 256
HEIGHT = 192
image_dataset = []
mask_dataset = []

# Load images and masks
masks = sorted(os.listdir(mask_directory))
for i, file_name in enumerate(masks):
    if file_name.endswith('tiff'):
        image = Image.open(image_directory + (file_name.replace('_finalprediction.ome', '')).replace('tiff', 'jpg'))
        mask = Image.open(mask_directory + file_name)
        mask = ImageOps.invert(mask)

        image.thumbnail((WIDTH, HEIGHT))
        mask.thumbnail((WIDTH, HEIGHT))

        image_pixels = asarray(image)
        image_pixels = image_pixels.astype('float32')
        image_pixels /= 255.0
        # image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2RGB) not needed anymore
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

# Split training and test set
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.1)


# -- Train --

# Load the model (and load weights)
# model = unet_model(HEIGHT, WIDTH, 3)
model = tensorflow.keras.models.load_model('models/keras')

# Format variables
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
