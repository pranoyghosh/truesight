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
from keras.models import model_from_json


print("[INFO] loading attributes...")
inputPath = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training.csv"
df = datasets.load_attributes(inputPath)

# load the object images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading images...")
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/training"
images = datasets.load_images(inputPath, img_data)
images = images / 255.0

print(images.shape)
print(df.shape)
# partition the data into training and testing splits using 95% of
# the data for training and the remaining 5% for validation
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.10, random_state=33)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
print(trainAttrX.shape)
# find the largest  bounding box coordinate for each x1 x2 y1 y2 in the training set and use it to
# scale our bounding box coordinates to the range [0, 1] (will lead to better
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

cnn = models.create_cnn(128, 128, 3, regress=False)


# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(cnn.output)
x1 = Dense(1, activation="linear")(x)
x2 = Dense(1, activation="linear")(x)
y1 = Dense(1, activation="linear")(x)
y2 = Dense(1, activation="linear")(x)

model = Model(inputs=cnn.input, outputs=[x1, x2, y1, y2])


opt = Adam(lr=1e-3, decay=1e-3 / 60)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[iou.mean_iou])

# train the model
print("[INFO] training model...")
model.fit(
    trainImagesX, [trainY1,trainY2,trainY3,trainY4],
    validation_data=(testImagesX, [testY1,testY2,testY3,testY4]),
    epochs=60, batch_size=6)

# evaluate the model
scores = model.evaluate(trainImagesX, [trainY1,trainY2,trainY3,trainY4], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model2.h5")
print("Saved model to disk")
