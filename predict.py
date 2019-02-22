# import the necessary packages
from module import datasets
from module import models
from keras.models import Model
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json

print("[INFO] loading attributes...")
trainPath = "/home/harshit1201/Desktop/Project:TrueSight/training_set.csv"
inputPath = "/home/harshit1201/Desktop/Project:TrueSight/test.csv"
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/images"
col1=["image_name"]
files=[]
df = pd.read_csv(inputPath, skiprows=[0], header=None, names=col1)
#df3 = df.drop(df.columns[[1, 2, 3, 4]], axis=1)
cols =["image_name","x1","x2","y1","y2"]
df2 = pd.read_csv(trainPath, skiprows=[0], header=None, names=cols)
for filename,p,n,e in df.index.values:
    files.append(os.path.join(img_data,filename))
# load the images and then scale the pixel intensities to the
# range [0, 1]
df = pd.read_csv(inputPath, skiprows=[0], header=None, names=cols)
genTCustom = datasets.custom_gentest(files,35)
# load json and create model
json_file = open('models/modelR3_5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/modelR3_5.h5")
print("Loaded model from disk")

# make predictions on the testing data
print("[INFO] predicting bounding boxes...")
preds = loaded_model.predict_generator(genTCustom,steps=687)
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

#preds = pd.DataFrame(preds, columns=['x1','x2','y1','y2']).to_csv('prediction.csv')
dfx = pd.DataFrame({'image_name' : df["image_name"], 'x1' : testY1, 'x2' : testY2, 'y1' : testY3, 'y2' : testY4})
dfx.to_csv("res/testR3_5.csv", index=False)

print('Predictions saved')
