import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras import layers


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


def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(24, (5,5), (2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(layers.Conv2D(36, (5,5), (2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(layers.Conv2D(48, (5,5), (2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(layers.Conv2D(64, (3,3),input_shape=(66, 200, 3), activation='elu'))          
    model.add(layers.Conv2D(64, (3,3),input_shape=(66, 200, 3), activation='elu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='elu'))
    model.add(layers.Dense(50, activation='elu'))
    model.add(layers.Dense(10, activation='elu'))
    model.add(layers.Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')
    return model


if __name__ == '__main__':
    data = DataInfo()
    data = balanceData(data)
    imgpath, steering = loadData('', data)
    x_train, x_test, y_train, y_test = train_test_split(imgpath, steering, test_size=0.2, random_state=5)
    model = create_model()
    history = model.fit(batch_gen(x_train, y_train, 100, True), steps_per_epoch=300, epochs=10, validation_data=batch_gen(x_test, y_test, 100, False), validation_steps=200, verbose=2)
    model.save('model.h5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    

