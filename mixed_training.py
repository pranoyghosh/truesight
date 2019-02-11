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


#num_classes = 1

#miou_metric = iou.MeanIoU(num_classes)
# construct the argument parser and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
help="path to input dataset of house images")
args = vars(ap.parse_args())
'''
# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading attributes...")
inputPath = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training.csv"
df = datasets.load_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training"
images = datasets.load_images(inputPath, img_data)
images = images / 255.0

print(images.shape)
print(df.shape)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.20, random_state=40)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
print(trainAttrX.shape)
# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxX1 = trainAttrX["x1"].max()
trainY1 = trainAttrX["x1"] / maxX1
testY1 = testAttrX["x1"] / maxX1
maxX2 = trainAttrX["x2"].max()
trainY2 = trainAttrX["x2"] / maxX2
testY2 = testAttrX["x2"] / maxX2
maxY1 = trainAttrX["y1"].max()
trainY3 = trainAttrX["y1"] / maxY1
testY3 = testAttrX["y1"] / maxY1
maxY2 = trainAttrX["y2"].max()
trainY4 = trainAttrX["y2"] / maxY2
testY4 = testAttrX["y2"] / maxY2

# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
#(trainAttrX, testAttrX) = datasets.process_house_attributes(df,
#	trainAttrX, testAttrX)

# create the MLP and CNN models
#mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(224, 224, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
#combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(cnn.output)
x1 = Dense(1, activation="linear")(x)
x2 = Dense(1, activation="linear")(x)
y1 = Dense(1, activation="linear")(x)
y2 = Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=cnn.input, outputs=[x1, x2, y1, y2])

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3 / 90)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[iou.mean_iou])

# train the model
print("[INFO] training model...")
model.fit(
    trainImagesX, [trainY1,trainY2,trainY3,trainY4],
    validation_data=(testImagesX, [testY1,testY2,testY3,testY4]),
    epochs=90, batch_size=8)

# evaluate the model
scores = model.evaluate(trainImagesX, [trainY1,trainY2,trainY3,trainY4], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model1.h5")
print("Saved model to disk")
