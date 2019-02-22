from module import datasets
from module import models
from keras.models import Model
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json

def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[2], boxB[2])
	xB = min(boxA[1], boxB[1])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
	boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

print("[INFO] loading attributes...")
iou=[]
trainPath = "/home/harshit1201/Desktop/Project:TrueSight/training_set.csv"
img_data = "/home/harshit1201/Desktop/Project:TrueSight/Dataset/images"
tarpath="eval/tarR3_7.csv"
col1=["image_name"]
files=[]
df = pd.read_csv(trainPath, skiprows=[0], header=None, names=col1)
#df3 = df.drop(df.columns[[1, 2, 3, 4]], axis=1)
cols =["image_name","x1","x2","y1","y2"]
df2 = pd.read_csv(trainPath, skiprows=[0], header=None, names=cols)
for filename,p,n,e in df.index.values:
    files.append(os.path.join(img_data,filename))
# load the images and then scale the pixel intensities to the
# range [0, 1]
genSCustom = datasets.custom_gentest(files,64)
# load json and create model
json_file = open('models/modelR3_7.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/modelR3_7.h5")
print("Loaded model from disk")

# make predictions on the testing data
print("[INFO] predicting bounding boxes...")
preds = loaded_model.predict_generator(genSCustom,steps=375)
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
dfx = pd.DataFrame({'image_name' : df2["image_name"], 'x1' : testY1, 'x2' : testY2, 'y1' : testY3, 'y2' : testY4})
for filename,p,n,e in df.index.values:
    boxA=datasets.load_attributes(df2,filename)
    boxB=datasets.load_attributes(dfx,filename)
    x=bb_iou(boxA, boxB)
    iou.append(x)

npiou=np.array(iou)
meanIOU=np.mean(npiou)
fdf1 = pd.DataFrame({'image_name' : df2["image_name"], 'iou' : npiou})
fdf2 = pd.DataFrame({'image_name' : ['mean_iou'], 'iou' : [meanIOU]})
fdf1 = fdf1.append(fdf2)
fdf1.to_csv(tarpath, index=False)

print('eval saved')
