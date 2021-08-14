import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.utils import shuffle

import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import os
import cv2
import random



def DataInfo():
    col = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv('driving_log.csv', names=col)
    data['Center'] = data['Center'].apply(lambda name: Path(name).name)
    data['Left'] = data['Left'].apply(lambda name: Path(name).name)
    data['Right'] = data['Right'].apply(lambda name: Path(name).name)
    return data



def balanceData(data, display=False):
    nBins = 31
    samplesPerBin = 500
    historgam, bins = np.histogram(data['Steering'], nBins)
    if display:
        
        center = (bins[:-1] + bins[1:]) * .5
        plt.bar(center, historgam, width = .06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()
    #remove data
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining', len(data))
    return data


def loadData(path,data):
    imgPath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i]
        imgPath.append(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imgPath = np.asarray(imgPath)
    steering = np.asarray(steering)
    return imgPath, steering


def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    if np.random.rand() < .5:
        pan = iaa.Affine(translate_percent={'x':(-.1, .1), 'y':(-.1, .1)})
        img = pan.augment_image(img)
    if np.random.rand() < .5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < .5:
        brightness = iaa.Multiply((.4, 1.2))
        img = brightness.augment_image(img)
    
    if np.random.rand() < .5:
        flip = cv2.flip(img, 1)
        steering = -steering

    return img, steering

def preProcessing(img):
    img = img[60: 135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255.0
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
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))




    
    

