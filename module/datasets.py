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

def load_images(inpdf, inputPath):
    # initialize our images array
    images = []
    cols =["image_name"]
    df = pd.read_csv(inpdf, skiprows=[0], header=None, names=cols)
    # loop over the indexes of the houses
    for filename,p,n,e in df.index.values:
        img = cv2.imread(os.path.join(inputPath,filename))
        c=cv2.resize(img, (128,128))
        #c=c/255
        images.append(c)
        # return our set of images
    return np.array(images)

def custom_genimg(files,df,bsize=8000):
    batch_paths=np.random.choice(a=files,size=bsize)
    batch_input=[]
    batch_output=[]
    for input_path in batch_paths:
        input=load_images(input_path)
        fname=os.path.basename(input_path)
        output=load_attributes(df,fname)
        input=preprocess_img(input)
        output=preprocess_att(output)
        batch_input+=[input]
        batch_output+=[output]
    batch_x=np.array(batch_input)
    batch_y=np.array(batch_output)
    yield(batch_x, batch_y)

def preprocess_att(inputPath, bsize=8000):
    cols = ["x1", "x2", "y1", "y2"]
    df = pd.read_csv(inputPath, skiprows=[0], header=None, names=cols)
    L=len(df)
    maxX1 = df["x1"].max()
    df["x1"] = df["x1"] / maxX1
    maxX2 = df["x2"].max()
    df["x2"] = df["x2"] / maxX2
    maxY1 = df["y1"].max()
    df["y1"] = df["y1"] / maxY1
    maxY2 = df["y2"].max()
    df["y2"] = df["y2"] / maxY2
    while True:
        bstart=0
        bend=bsize
        while bstart < L:
            df=df.values[bstart:bend]
            # return the data frame
            return df
        bstart+=bsize
        bend+=bsize



def preprocess_img(inpdf, inputPath, bsize=8000):
    cols =["image_name"]
    df = pd.read_csv(inpdf, skiprows=[0], header=None, names=cols)
    L=len(df)
    while True:
        # initialize our images array

        bstart=0
        bend=bsize
        while bstart < L:
            images = []
            # loop over the indexes of the houses
            for filename,p,n,e in df.index.values[bstart:bend]:
                img = cv2.imread(os.path.join(inputPath,filename))
                c=cv2.resize(img, (128,128))
                c=c/255
                images.append(c)
                # return our set of images
            return images
        bstart+=bsize
        bend+=bsize
