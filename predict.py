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
import pandas as pd
import locale
import os
from keras.models import model_from_json

print("[INFO] loading attributes...")
trainPath = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training.csv"
inputPath = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/test.csv"
cols =["image_name"]
df = pd.read_csv(inputPath, skiprows=[0], header=None, names=cols)
df2 = datasets.load_attributes(trainPath)
# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/test"
images = datasets.load_images(inputPath, img_data)
images = images / 255.0


testImagesX = images
# load json and create model
json_file = open('models/model4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model4.h5")
print("Loaded model from disk")

# make predictions on the testing data
print("[INFO] predicting bounding boxes...")
preds = loaded_model.predict(testImagesX)
testY1 = preds[0]
testY2 = preds[1]
testY3 = preds[2]
testY4 = preds[3]
testY1 = testY1.flatten()
testY2 = testY2.flatten()
testY3 = testY3.flatten()
testY4 = testY4.flatten()
print(testY1.shape)

maxX1 = df2["x1"].max()
testY1 = testY1 * maxX1
maxX2 = df2["x2"].max()
testY2 = testY2 * maxX2
maxY1 = df2["y1"].max()
testY3 = testY3 * maxY1
maxY2 = df2["y2"].max()
testY4 = testY4 * maxY2
imgcol=df["image_name"]
#preds = pd.DataFrame(preds, columns=['x1','x2','y1','y2']).to_csv('prediction.csv')
dfx = pd.DataFrame({'image_name' : imgcol, 'x1' : testY1, 'x2' : testY2, 'y1' : testY3, 'y2' : testY4})
dfx.to_csv("res/test.csv")

print('Predictions saved')
