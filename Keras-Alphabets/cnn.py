import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D

train_dest = './Alphabets/train.csv'
test_dest = './Alphabets/test.csv'

def read_csv(filename):
    x = []
    y = []
    f = open(filename, 'r')
    for xx in f:
        l = xx.split(',')
        x.append([float(i)/255 for i in l[:784]])
        y.append([int(l[784])])
    return np.array(x), keras.utils.to_categorical(np.array(y), num_classes=26, dtype='float32')

x_train, y_train = read_csv(train_dest)
x_test, y_test = read_csv(test_dest)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_val = x_train[-3000:]
y_val = y_train[-3000:]
x_train = x_train[:-3000]
y_train = y_train[:-3000]

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape = (28,28,1) ))
model.add(MaxPooling2D(pool_size=2, strides=None, data_format='channels_last'))
# model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(26, activation = 'softmax'))

print(model.summary())
optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=100, epochs = 20)

score = model.evaluate(x_test, y_test, batch_size=128)
print('Test accuracy: ', score[-1])

hist = model.history.history

# Plotting loss function over epochs
x_axis = np.arange(1, (len(hist['loss']))+1)
plt.title('Loss v/s epochs')
plt.plot(x_axis, hist['loss'], label='loss')
plt.grid()
plt.legend(loc='best')
plt.show()
plt.clf()
