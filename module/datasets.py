# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_attributes(inputPath):
    cols = ["x1", "x2", "y1", "y2"]
    df = pd.read_csv(inputPath, header=None, names=cols)
    # return the data frame
    return df


def load_images(inputPath):
    # initialize our images array
    images = []
    # loop over the indexes of the houses
    for filename in os.listdir(inputPath):
        img = cv2.imread(os.path.join(inputPath,filename))
        c=cv2.resize(img, (64,64))
        images.append(c)
        # return our set of images
    return np.array(images)
