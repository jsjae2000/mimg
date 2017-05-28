'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sL290A_resize_load.py
'''

#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from numpy import genfromtxt

import os
import time

from PIL import Image

#### input
## resize input
#delete / at windows
path_image = "/home/nai/sL290A" # location of original images
out_image = "/home/nai/sL290A_144" # save converted images at out_image
img_rows, img_cols = 144, 144

## load and save images input
load_path = "/home/nai/sL290A_144"
rate_sampling = 1 ########
out_data = "/home/nai/CNN_SJ/data_file/sL290A_144.bin" # data file
nb_classes = 2
norm_tp = "ACSP1"
rate_train = 0.6

## load and save feature data input
# load_path
feat_data_name = 'L290A_df.csv'
not_numeric_column = [0, 5, 6, 9]
image_name_column = 2 # in not_numeric_column





###### resize
##folder_image = os.listdir(path_image)
##
##tt = time.time()
##
##i=0
##for image_type in folder_image:
##    for image in os.listdir(path_image + "/" + image_type):
##        if i%10000 == 0: print "Resizing %dth image" %i
##        img = Image.open(path_image + "/" + image_type + "/" + image)
##        img = img.resize((img_cols, img_rows), Image.ANTIALIAS)
##        img.save(out_image + "/" + image_type + "/" + image)
##        i+=1
##        
##print time.time() - tt





#### load and save images

## preparation
folder_image = os.listdir(load_path)

n=0
for image_type in folder_image:
    if image_type == norm_tp:
        from_norm_tp = n
        count_norm_tp = len( os.listdir(load_path + "/" + image_type) )
    n += len( os.listdir(load_path + "/" + image_type) )

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
    for image in os.listdir(load_path + "/" + image_type):
        if i in mysample:
            img[j_sample,:,:,:] = mpimg.imread(load_path + "/" + image_type + "/" + image)
            j_sample += 1
            if j_sample%10000==0: print "Loading %d images" %j_sample
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

print('X_train shape:', X_train.shape)
print('Y_train:', sum(Y_train[:,0]), sum(Y_train[:,1]))

print('X_test shape:', X_test.shape)
print('Y_test:', sum(Y_test[:,0]), sum(Y_test[:,1]))



#### load and save feature data - use training / test and shuffling above
## load feature data
char_data = genfromtxt(feat_data_name, usecols=not_numeric_column ,skip_header=True, dtype=np.str, delimiter=',')
feat_data = genfromtxt(feat_data_name, skip_header=True, dtype='float32', delimiter=',')
feat_data = np.delete(feat_data, not_numeric_column, 1)
print 'character in feature data:', char_data.shape
print 'additional feature data:', feat_data.shape

## sort
feat_n = feat_data.shape[0]
folder_image = os.listdir(load_path)
_order_ = np.zeros( feat_n, dtype=int )

sort_t = time.time()
i=0
for image_type in folder_image:
    for image in os.listdir(load_path + "/" + image_type):
        if i%10000 == 0: print "Sorting %dth data" %i
        _order_[i] = np.argmax( char_data[:,image_name_column] == '"'+image[0:-4]+'"' )
        i += 1   
print time.time() - sort_t

if sum( abs( np.arange(feat_n)-np.sort(_order_) ) )==0: print 'sorting feature data: good'

feat_sort = feat_data[_order_,:]

## training / test
feat_train = feat_sort[sample_train,:]
feat_test = np.delete(feat_sort, sample_train, 0)

## shuffling
feat_train = feat_train[shuffle_train,:]
feat_test = feat_test[shuffle_test,:]
print 'feat_train:', feat_train.shape
print 'feat_test:', feat_test.shape



#### save data
outdata = file(out_data, "wb")
np.savez(outdata, X_train = X_train, Y_train = Y_train, feat_train = feat_train, X_test = X_test, Y_test = Y_test, feat_test = feat_test)
outdata.close()
