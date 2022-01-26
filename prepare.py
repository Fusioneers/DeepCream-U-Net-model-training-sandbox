import cv2
import os
import numpy as np


def clear_directory(directory):
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def merge_red_blue_green(images_directory="data/38-Cloud_training/train_red",
                         output="data/generated/generated_patches/images"):
    # Goes through images in images_directory and merges red, blue and green pictures from their folders, then copies
    # them to output
    for image_file in os.listdir(images_directory):
        if not image_file:
            return
        filename = image_file
        if filename.endswith(".TIF"):
            # The photos must be named red_patch_anything_else.TIF, green_patch_anything_else.TIF and
            # blue_patch_anything_else.TIF, the folders must be named train_red, train_blue and train_green
            red = cv2.imread(str(os.path.join(images_directory, filename)))
            green = cv2.imread(
                str(str(os.path.join(images_directory, filename)).replace('train_red', 'train_green')).replace(
                    'red_patch', 'green_patch'))
            blue = cv2.imread(
                str(str(os.path.join(images_directory, filename)).replace('train_red', 'train_blue')).replace(
                    'red_patch', 'blue_patch'))

            red = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
            green = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
            blue = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

            image = cv2.merge((blue, green, red))

            # The output files will be named train_anything_else.TIF
            cv2.imwrite(os.path.join(output, filename.replace("red_patch", "train")),
                        image)


def copy_ground_truth(gt_directory="data/38-Cloud_training/train_gt", output="data/generated/generated_patches/masks"):
    # Goes through masks in gt_directory and copies them to output as train_
    for gt_file in os.listdir(gt_directory):
        if not gt_file:
            return
        filename = gt_file
        if filename.endswith(".TIF"):
            image = cv2.imread(str(os.path.join(gt_directory, filename)))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output, filename.replace("gt_patch", "train")), gray)


def clean_directory(image_directory="data/generated/generated_patches/images",
                    mask_directory="data/generated/generated_patches/masks"):
    # Goes through the image directory and deletes all images that have more than 20 black pixels, also deletes the
    # mask file with the same name from the mask directory
    for file in os.listdir(image_directory):
        filename = file
        if filename.endswith(".TIF"):
            image = cv2.imread(str(os.path.join(image_directory, filename)))

            if np.sum(image == 0) > 20:
                os.remove(str(os.path.join(mask_directory, filename)))
                os.remove(str(os.path.join(image_directory, filename)))


# print("Clearing directories")
# clear_directory("data/generated/generated_patches/images")
# clear_directory("data/generated/generated_patches/masks")
# print("Merging red, blue and green channels")
# merge_red_blue_green()
# print("Copying ground truth")
# copy_ground_truth()
# print("Cleaning bad training data")
# clean_directory()
