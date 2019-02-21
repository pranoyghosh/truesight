# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_attributes(df,fname):
    out=df.loc[df["image_name"]==fname]
    del out["image_name"]
    out=out.values.tolist()
    # return the data frame
    return out

def load_images(inputPath):
    img = cv2.imread(inputPath)
    c=cv2.resize(img, (128,128))
    return c

def preprocess_att(df,inp):
    maxX1 = df["x1"].max()
    inp[0][0] = inp[0][0] / maxX1
    maxX2 = df["x2"].max()
    inp[0][1] = inp[0][1] / maxX2
    maxY1 = df["y1"].max()
    inp[0][2] = inp[0][2] / maxY1
    maxY2 = df["y2"].max()
    inp[0][3] = inp[0][3] / maxY2
    return inp

def preprocess_img(inpimg):
    img=inpimg/255
    return img

def custom_genimg(files,df,bsize=8000):
    while True:
        batch_paths=np.random.choice(a=files,size=bsize)
        batch_input=[]
        batch_output=[]
        for input_path in batch_paths:
            input=load_images(input_path)
            fname=os.path.basename(input_path)
            output=load_attributes(df,fname)
            input=preprocess_img(input)
            output=preprocess_att(df,output)
            batch_input+=[input]
            batch_output+=[output]
        batch_x=np.array(batch_input)
        batch_y=np.array(batch_output)
        yield(batch_x, batch_y)
