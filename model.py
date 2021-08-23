import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.applications import ResNet50



def create_resnet():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    for layer in resnet.layers[:-4]:
        layer.trainable = False
 
     return resnet

    
def create_model():
  resnet = create_resnet()
  
  model = tf.keras.Sequential()
  model.add(resnet)

  model.add(layers.Dropout(0.5))
  
  model.add(layers.Flatten())
  
  model.add(layers.Dense(100, activation='elu'))
  model.add(layers.Dropout(0.5))
  
  model.add(layers.Dense(50, activation='elu'))
  model.add(layers.Dropout(0.5))
  
  model.add(layers.Dense(10, activation='elu'))
  model.add(layers.Dropout(0.5))
  
  model.add(layers.Dense(1))
  
  optimizer = tf.keras.optimizers.Adam(lr=1e-3)
  model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
  return model

def graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
