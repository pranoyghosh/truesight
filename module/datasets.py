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
    out=out[0]
    # return the data frame
    return out

def load_images(inputPath):
    img = cv2.imread(inputPath)
    c=cv2.resize(img, (288,288))
    return c

def preprocess_att(df,inp,i):
    if i==0:
        maxCor = df["x1"].max()
        inp[i] = inp[i] / maxCor
    elif i==1:
        maxCor = df["x2"].max()
        inp[i] = inp[i] / maxCor
    elif i==2:
        maxCor = df["y1"].max()
        inp[i] = inp[i] / maxCor
    elif i==3:
        maxCor = df["y2"].max()
        inp[i] = inp[i] / maxCor
    return inp[i]

def preprocess_img(inpimg):
    img=inpimg/255.0
    return img

def custom_genimg(files,df,bsize=6):
    n=0
    while True:
        if n>=24000:
            n=0
        batch_paths=files[n:n+bsize]
        #batch_paths=np.random.choice(a=files,size=bsize)
        batch_input=[]
        batch_outputx1=[]
        batch_outputx2=[]
        batch_outputy1=[]
        batch_outputy2=[]
        for input_path in batch_paths:
            input=load_images(input_path)
            fname=os.path.basename(input_path)
            output=load_attributes(df,fname)
            input=preprocess_img(input)
            outputx1=preprocess_att(df,output,0)
            outputx2=preprocess_att(df,output,1)
            outputy1=preprocess_att(df,output,2)
            outputy2=preprocess_att(df,output,3)
            batch_input+=[input]
            batch_outputx1+=[outputx1]
            batch_outputx2+=[outputx2]
            batch_outputy1+=[outputy1]
            batch_outputy2+=[outputy2]
        batch_x=np.array(batch_input)
        batch_tx1=np.array(batch_outputx1)
        batch_tx2=np.array(batch_outputx2)
        batch_ty1=np.array(batch_outputy1)
        batch_ty2=np.array(batch_outputy2)
        #batch_y=batch_y.flatten()
        yield (batch_x, {'op1':batch_tx1, 'op2':batch_tx2, 'op3':batch_ty1, 'op4':batch_ty2})
        n+=bsize

def custom_gentest(files,bsize=6):
    n=0
    while True:
        batch_paths=files[n:n+bsize]
        batch_input=[]
        for input_path in batch_paths:
            input=load_images(input_path)
            fname=os.path.basename(input_path)
            input=preprocess_img(input)
            batch_input+=[input]
        batch_x=np.array(batch_input)
        print(n)
        #batch_y=batch_y.flatten()
        yield (batch_x)
        n+=bsize
