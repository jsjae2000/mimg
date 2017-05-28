import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import os
import time; 
from tempfile import TemporaryFile

from PIL import Image

#### input
## resize input
#delete / at windows
path_image = "/home/nai/L691A" # location of original images
out_image = "/home/nai/L691A_224" # save converted images at out_image
img_rows = 224 ########
img_cols = 224 ########

## load and save images input
rate_sampling = 1 ########
out_data = "L691A_224.bin" # data file
nb_classes = 2
norm_tp = "SDSP1"
rate_train = 0.6



###### resize
##folder_image = os.listdir(path_image)
##
##tt = time.time()
##
##i=0
##print 'the number of image types:', len(folder_image)
##for image_type in folder_image:
##    print 'Resizing', image_type, 'images'
##    if image_type == 'ACDLS': continue
##    if image_type == 'PXXU1': continue
##    if image_type == 'ACSR2': continue
##    if image_type == 'ACSR5': continue
##    if image_type == 'ACSR3': continue
##    if image_type == 'SDSLS': continue
##    if image_type == 'PXXE2': continue
##    if image_type == 'SDSR2': continue
##    if image_type == 'SDSR3': continue
##    if image_type == 'SDSP1': continue
##    if image_type == 'NGSLS': continue
##    if image_type == 'PXXU2': continue
##    if image_type == 'NGSR1': continue
##    if image_type == 'SDSRM': continue
##    if image_type == 'ILDLS': continue
##    if image_type == 'SDSR5': continue
##    if image_type == 'PXXNF': continue
##    if image_type == 'ILPR1': continue
##    for image in os.listdir(path_image + "/" + image_type):
##        #if i%5000 == 0: print "Resizing %dth image" %i
##        img = Image.open(path_image + "/" + image_type + "/" + image)
##        img = img.resize((img_cols, img_rows), Image.ANTIALIAS)
##        img.save(out_image + "/" + image_type + "/" + image)
##        i+=1
##        
##print time.time() - tt



#### load and save images

## preparation
path_image = out_image
folder_image = os.listdir(path_image)

n=0
for image_type in folder_image:
    if image_type == norm_tp:
        from_norm_tp = n
        count_norm_tp = len( os.listdir(path_image + "/" + image_type) )
    n += len( os.listdir(path_image + "/" + image_type) )

## sampling
np.random.seed(1030)
n_sampling = round(n*rate_sampling)
mysample = np.random.choice(n, n_sampling, replace=False)
mysample = np.sort(mysample)


## load images
load_t = time.time()

i = j_sample = 0
img = np.zeros([n_sampling, img_rows, img_cols, 3])
for image_type in folder_image:
    for image in os.listdir(path_image + "/" + image_type):
        if i in mysample:
            img[j_sample,:,:,:] = mpimg.imread(path_image + "/" + image_type + "/" + image)
            j_sample += 1
            if j_sample%5000==0: print "Loading %d images" %j_sample
        i += 1
img = img.transpose(0,3,1,2)

print time.time() - load_t

## save y_data
y_data = np.ones(n)
y_data[from_norm_tp:(from_norm_tp+count_norm_tp)] = 0
y_data = y_data[mysample]

## separate training / test data
np.random.seed(2016)
n_train = round(img.shape[0]*rate_train)
sample_train = np.random.choice(img.shape[0], n_train, replace=False)
sample_train = np.sort(sample_train)

X_train = img[sample_train,:,:,:]/255
X_test = np.delete(img, sample_train, 0)/255
y_train = y_data[sample_train]
y_test = np.delete(y_data, sample_train, 0)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

## shuffle training / test data
np.random.seed(1234)
shuffle_train = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
shuffle_test = np.random.choice(X_test.shape[0], X_test.shape[0], replace=False)

X_train = X_train[shuffle_train,:,:,:]
Y_train = Y_train[shuffle_train,:]
X_test = X_test[shuffle_test,:,:,:]
Y_test = Y_test[shuffle_test,:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print 'X_train shape:', X_train.shape
print 'Y_train:', sum(Y_train[:,0]), sum(Y_train[:,1])

print 'X_test shape:', X_test.shape
print 'Y_test:', sum(Y_test[:,0]), sum(Y_test[:,1])

#### save data
outdata = file(out_data, "wb")
np.savez(outdata, X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
outdata.close()
