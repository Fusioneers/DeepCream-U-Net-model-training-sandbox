# U-Net model training sandbox

This is the project we used to train the machine learning model used in the cloud detection part of DeepCream.
It features a simple U-Net model, a script to train and one to test it. Also included is a script to convert the
keras model to a tensorflow lite model (.tflite) and test that model on a tpu (like Google coral tpu). So all in all
it's a nice starting point for an AI project, especially if you are just starting out with machine development and
are looking for some hands-on projects where you can get started quickly without too much prior knowledge.

## How to use it
Here's a simple rundown of when and how to use this project.

1. Decide if a U-Net model is the right architecture for your purposes
2. Collect training data
3. Download the necessary python libraries (see requirements.txt)
4. Recreate the project structure see below **(GitHub doesn't allow for empty folders to be uploaded, so you will have to create the data, images, masks, models, keras and tflite folders!)**
5. Format the training data to the models needs
   1. The model expects a folder "data" with two folders inside called
   "images" and "masks".
   2. The corresponding mask and image files should be named identically and put in the respective directories
   3. The images should be in the ".jpg" format and the
   masks in the ".tiff" format (change the file_type variable in the script if that's not the case).
5. Run the train.py to train your model, the result will be visible in the models/keras folder
6. Run the test.py to test your model

## Project structure
 U-Net-model-training-sandbox/
├─ data/
│  ├─ images/
│  │  ├─ example1.jpg
│  │  ├─ example2.jpg
│  │  ├─ ...
│  ├─ masks/
│  │  ├─ example1.tiff
│  │  ├─ example2.tiff
│  │  ├─ ...
├─ models/
│  ├─ keras/
│  ├─ tflite/
├─ .gitignore/
├─ README.md
├─ convert_to_tflite.py
├─ requirements.txt
├─ test.py
├─ test_tpu.py
├─ train.py
├─ unet_model.py

## Contribute
If you discover a bug or have a suggestion on how to improve this sandbox, feel free to open an issue or
contact us directly.

## Useful links ...
### ... to learn about U-Net models  
https://en.wikipedia.org/wiki/U-Net  
https://developers.arcgis.com/python/guide/how-unet-works/

### ... to collect some training data
https://www.kaggle.com/

### ... to get into image segmentation without any coding abilities 
https://www.apeer.com/home/
