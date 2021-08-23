import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.utils import shuffle

import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import os
import cv2
import random



def DataImport():
    col = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv('driving_log2.csv', names=col)
    data['Center'] = data['Center'].apply(lambda name: name.split('\\')[-1])
    #print(data.shape)
    return data



def balanceData(data, display=False):
    size = 1000
    nbins = 31
    
    _, bins = pd.cut(data['Steering'], nbins, retbins=True)
    
    if display:
        plt.hist(data['Steering'], nbins)
        plt.plot((-1,1), (size, size))
        plt.show()
    
    removeList = []
    for i in range(nbins):
        binList = []
        for ind, d in data.iterrows():
            if d.Steering >= bins[i] and d.Steering < bins[i+1]:
                binList.append(ind)
        random.shuffle(binList)
        removeList.extend(binList[size:])
    data.drop(data.index[removeList], inplace=True)
    
    if display:
        plt.hist(data['Steering'], nbins)
        plt.plot((-1,1), (size, size))
        plt.show()
    return data


def loadData(path,data):
    imgPath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i]
        imgPath.append(Path(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imgPath, steering = np.asarray(imgPath), np.asarray(steering)
    return imgPath, steering


def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    if np.random.rand() < .5:
        img = translate(img)
    if np.random.rand() < .5:
        img = zoom(img)
    
    if np.random.rand() < .5:
        img = brightness(img)
    
    if np.random.rand() < .5:
        img, steering = flip(img, steering)
    
    if np.random.rand() < .5:
        img = tilt(img)
    return img, steering



def translate(img):
    x = random.random() * .2 - .05
    y = random.random() * .2 - .05
    M= np.float32([[1, 0, x], [0, 1, y]])
    height, width = img.shape[:2]
    image = cv2.warpAffine(img, M, (width, height))
    return image

def zoom(img):
    x = 1 + random.random() * .2
    image = cv2.resize(img,None,fx=x, fy=x, interpolation = cv2.INTER_LINEAR)
    return image

def brightness(img):
    brightness = iaa.Multiply((.4, 1.2))
    image = brightness.augment_image(img)
    return image

def flip(img, steering):
    image = cv2.flip(img, 1)
    return image, -steering

def tilt(img):
    angle = random.random()*2 - 1
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(img, M, (w, h))
    return image


def preProcessing(img):
    img = img[60: 135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255
    return img

def batch_gen(path, steeringList, batch, train):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batch):
            index = random.randint(0, len(path)-1)
        
            if train:
                img, steering = augmentImage(path[index], steeringList[index])
            else:
                img = mpimg.imread(path[index])
                steering = steeringList[index]
            
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        imgBatch, steeringBatch = np.asarray(imgBatch), np.asarray(steeringBatch)
        yield imgBatch, steeringBatch




    
    

