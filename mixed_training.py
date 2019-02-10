# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/

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

num_classes = 1

miou_metric = iou.MeanIoU(num_classes)
# construct the argument parser and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())
'''
# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = "/media/pranoy/New Volume1/truesight/training.csv"
df = datasets.load_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
img_data = "/media/pranoy/New Volume1/truesight/train"
images = datasets.load_images(img_data)
images = images / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

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
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x1 = Dense(1, activation="linear")(x)
x2 = Dense(1, activation="linear")(x)
y1 = Dense(1, activation="linear")(x)
y2 = Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=[x1,x2,y1,y2])

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
#opt = Adam(lr=1e-3, decay=1e-3 / 200)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(optimizer='adam', loss="mean_absolute_percentage_error", metrics=[miou_metric.mean_iou])

# train the model
print("[INFO] training model...")
model.fit(
	[trainAttrX, trainImagesX], [trainY1,trainY2,trainY3,trainY4],
	validation_data=([testAttrX, testImagesX], [testY1,testY2,testY3,testY4]),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])
'''
# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
'''
