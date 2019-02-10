# import the necessary packages
from module import datasets
from module import models
from module import iou
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
#import argparse
import locale
import os
from keras.models import model_from_json

print("[INFO] loading attributes...")
inputPath = "/media/pranoy/New Volume1/truesight/data/test.csv"
df = datasets.load_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
img_data = "/media/pranoy/New Volume1/truesight/data/test"
images = datasets.load_images(img_data)
images = images / 255.0

testAttrX = df
testImagesX = images
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# make predictions on the testing data
print("[INFO] predicting bounding boxes...")
preds = loaded_model.predict([testAttrX, testImagesX])
preds = pd.DataFrame(preds, columns=['x1','x2','y1','y2']).to_csv('prediction.csv')
