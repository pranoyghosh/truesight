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

cnn = models.create_cnn(128, 128, 3, regress=False)


# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(cnn.output)
x1 = Dense(1, activation="linear")(x)
x2 = Dense(1, activation="linear")(x)
y1 = Dense(1, activation="linear")(x)
y2 = Dense(1, activation="linear")(x)
combinedOutput = concatenate([x1,x2,y1,y2])

model = Model(inputs=cnn.input, outputs=combinedOutput)

genCustom = datasets.custom_genimg(files,df,6)
opt = Adam(lr=1e-3, decay=1e-3 / 60)
#model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[iou.mean_iou])

# train the model
print("[INFO] training model...")
model.fit_generator(
    genCustom,
    epochs=60, steps_per_epoch=4000)

# evaluate the model
#scores = model.evaluate(trainImagesX, [trainY1,trainY2,trainY3,trainY4], verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")
