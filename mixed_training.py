# import the necessary packages
import tensorflow as tf
from module import datasets
from module import models
from module import iou
from keras import optimizers
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

print("[INFO] processing data...")

cnn = models.create_cnn(256, 256, 3, regress=False)


# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(8, activation="relu")(cnn.output)
x = Dense(4, activation="relu")(x)
x1 = Dense(1, activation="linear", name='op1')(x)

x2 = Dense(1, activation="linear", name='op2')(x)

y1 = Dense(1, activation="linear", name='op3')(x)

y2 = Dense(1, activation="linear", name='op4')(x)


model = Model(inputs=cnn.input, outputs=[x1,x2,y1,y2])

genCustom = datasets.custom_genimg(files,df,32)
#opt = Adam(lr=1e-3, decay=1e-3 / 40)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
sgd = optimizers.SGD(lr=1e-3, decay=1e-3/40, momentum=0.8, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics={'op1':iou.mean_iou, 'op2':iou.mean_iou, 'op3':iou.mean_iou,'op4':iou.mean_iou})

# train the model
print("[INFO] training model...")
model.fit_generator(
    genCustom,
    epochs=40, steps_per_epoch=750)

# evaluate the model
#scores = model.evaluate_generator(genCustom, steps=750, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/modelR3_4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/modelR3_4.h5")
print("Saved model to disk")
