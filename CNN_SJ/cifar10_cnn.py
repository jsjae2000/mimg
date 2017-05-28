'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import json

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

batch_size = 128
nb_classes = 10
nb_epoch = 80

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
# number of convolutional filters to use
nb_filters = 5
# convolution kernel size
nb_conv = 5
# size of pooling area for max pooling
nb_pool = 2

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('sigmoid'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

hist = model.fit(X_train, Y_train, batch_size=batch_size, 
                 nb_epoch=nb_epoch, show_accuracy=True, 
				 verbose=1, validation_split=0.2)
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Save the model and weights
json_string = model.to_json()
open('cifar10_model_architecture.json','w').write(json_string)
model.save_weights('cifar10_model_weights.h5')

# Save History
with open('cifar10_model_history.json','w') as fp:
    json.dump(hist.history, fp)

