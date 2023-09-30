"""
The most important file for Grand-Challenge Algorithm submission is this process.py.
This is the file where you will extend our base algorithm class,
and modify the subclass of MyCoWSegAlgorithm for your awesome algorithm :)
Simply update the TODO in this file.

NOTE: remember to COPY your required files in your Dockerfile
COPY --chown=user:user <somefile> /opt/app/
"""

# Import files and defined functions

from data_preprocessing import data_preprocessing_function, concatenate_xyz_and_labels, create_image_data_dict, batch_data
from data_preprocessing import load_image, predictions
from base_algorithm import TASK, TRACK, BaseAlgorithm

# Used packages 
import keras
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
import random
from sklearn.preprocessing import LabelEncoder
import SimpleITK as sitk
import os
import json
from medpy.io import load as medpyload
import random
import pandas as pd
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import torch 
import h5py


#######################################################################################
track = TRACK.CT
task = TASK.MULTICLASS_SEGMENTATION
#######################################################################################


class MyCoWSegAlgorithm(BaseAlgorithm):

    def __init__(self):
        super().__init__(
            track=track,
            task=task,
        )

        # Load the model from the .h5 file
        # model1 = tf.keras.models.load_model('model_1_with_num_class_13.h5') # tensorflow version 
        # model2 = tf.keras.models.load_model('model_2_with_num_class_16.h5') 
        # self.model = model1.load_weights('best_model_weights.h5')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch version 
        # Check if GPU is available, and set the device accordingly
        # self.device = tf.device('GPU' if tf.config.experimental.list_physical_devices('GPU') else 'CPU') # tensorflow version
        
    

    def predict(self, *, image_ct: str, image_mr: str) -> np.array:
        """
        Inputs will be a pair of CT and MR .mha SimpleITK.Images
        Output is supposed to be an numpy array in (x,y,z)
        """
        print("image_ct:", image_ct)

        print(type(image_ct), "before preprocessing")
        image_data = load_image(image_ct)

        print("Before model")
        model = tf.keras.models.load_model('model_13.h5') # tensorflow version 
        print("After loading model")

        model = model.load_weights('best_model_13_weights.h5')
        print("After loading weights")
        model_weights_path = 'best_model_13_weights.h5'
    
        # Call the predict_labels method to get predictions
        print("Before predictions")
        predicted_output = predictions(image_data, model_weights_path, model)
        # Create a placeholder array for the prediction
        pred_array = np.array(predicted_output).reshape(image_ct.GetSize()[2], image_ct.GetSize()[1], image_ct.GetSize()[0]) 
        # reorder from (z,y,x) to (x,y,z)
        pred_array = pred_array.transpose((2, 1, 0)).astype(np.uint8)

        # return prediction array
        return pred_array


if __name__ == "__main__":
    # NOTE: running locally ($ python3 process.py) has advantage of faster debugging
    # but please ensure the docker environment also works before submitting
    MyCoWSegAlgorithm().process()
    cowsay_msg = """\n
  ____________________________________
< MyCoWSegAlgorithm().process()  Done! >
  ------------------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
    """
    print(cowsay_msg)
