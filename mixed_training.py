# import the necessary packages
import tensorflow as tf
from module import datasets
from module import models
from module import iou
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.layers import Flatten
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
'''
print("[INFO] loading attributes...")
files=[]
inputPath = "/home/harshit1201/Desktop/Project:TrueSight/training_set.csv"
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/images"
cols =["image_name"]
fdf = pd.read_csv(inputPath, skiprows=[0], header=None, names=cols)
for filename,p,n,e in fdf.index.values:
    files.append(os.path.join(img_data,filename))
colsdf = ["image_name","x1", "x2", "y1", "y2"]
df = pd.read_csv(inputPath, skiprows=[0], header=None, names=colsdf)

#images = datasets.load_images(inputPath, img_data)
#images = images / 255.0

#print(images.shape)
#print(df.shape)
# partition the data into training and testing splits using 95% of
# the data for training and the remaining 5% for validation
print("[INFO] processing data...")

cnn = models.create_cnn(224, 224, 3, regress=False)


# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(cnn.output)
#x = Flatten()(x)
x1 = Dense(1, activation="linear", name='op1')(x)
#x1 = Flatten()(x1)
x2 = Dense(1, activation="linear", name='op2')(x)
#x2 = Flatten()(x2)
y1 = Dense(1, activation="linear", name='op3')(x)
#y1 = Flatten()(y1)
y2 = Dense(1, activation="linear", name='op4')(x)
#y2 = Flatten()(y2)
#z=np.array([x1,x2,y1,y2])
#combinedOutput = concatenate([x1,x2,y1,y2])
#print(combinedOutput.shape)
#combinedOutput = Flatten()(combinedOutput)

model = Model(inputs=cnn.input, outputs=[x1,x2,y1,y2])

genCustom = datasets.custom_genimg(files,df,6)
opt = Adam(lr=1e-3, decay=1e-3 / 55)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[iou.mean_iou])

# train the model
print("[INFO] training model...")
model.fit_generator(
    genCustom,
    epochs=55, steps_per_epoch=4000)

# evaluate the model
#scores = model.evaluate(trainImagesX, [trainY1,trainY2,trainY3,trainY4], verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/modelR3_2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/modelR3_2.h5")
print("Saved model to disk")
