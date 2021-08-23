import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from selfdriving import *
from model import *
import datetime




data = DataImport()
data = balanceData(data, display=True)
imgpath, steering = loadData('', data)
x_train, x_test, y_train, y_test = train_test_split(imgpath, steering, test_size=0.2, random_state=5)
model = create_model()
history = model.fit(batch_gen(x_train, y_train, 100, True), steps_per_epoch=300, epochs=10, validation_data=batch_gen(x_test, y_test, 100, False), validation_steps=200, verbose=1)
model.save('model.h5')
print("saved")
graph(history)


