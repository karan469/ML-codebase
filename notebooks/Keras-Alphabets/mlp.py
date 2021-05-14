import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout

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

x_val = x_train[-3000:]
y_val = y_train[-3000:]
x_train = x_train[:-3000]
y_train = y_train[:-3000]

model = Sequential([
    Dense(100, input_dim=784),
    Activation('relu'),
    Dropout(0.3),
    Dense(26),
    Activation('softmax'),
])

print(model.summary())

# Compiling Model created
# sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) #Stochastic gradient descent
rms = optimizers.RMSprop(learning_rate=0.01, rho=0.9) #rms descent algorithm
model.compile(loss='mean_squared_error', optimizer=rms, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=100)

score = model.evaluate(x_test, y_test, batch_size=128)
print('Test accuracy: ', score[-1])

hist = model.history.history

# Plotting loss function over epochs
# x_axis = np.arange(1, (len(hist['loss']))+1)
# plt.title('Loss v/s epochs')
# plt.plot(x_axis, hist['loss'], label='loss')
# plt.grid()
# plt.legend(loc='best')
# plt.show()
# plt.clf()
