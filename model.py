import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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

def graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
