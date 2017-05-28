'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sL290A_v02_combine.py
'''

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import os
import time
from tempfile import TemporaryFile

from numpy import genfromtxt

#### input
feat_data_name = '/home/nai/CNN_SJ/L290A_df.csv'
path_image = "/home/nai/sL290A"
rate_train = 0.6
not_numeric_column = [0, 5, 6, 9]
image_name_column = 2 # in not_numeric_column
out_data = "sL290A_feature.bin"



#### load feature data
## load csv data
char_data = genfromtxt(feat_data_name, usecols=not_numeric_column ,skip_header=True, dtype=np.str, delimiter=',')
feat_data = genfromtxt(feat_data_name, skip_header=True, dtype='float32', delimiter=',')
feat_data = np.delete(feat_data, not_numeric_column, 1)
print 'character data:', char_data.shape
print 'additional feature data:', feat_data.shape

## sort
n = feat_data.shape[0]
folder_image = os.listdir(path_image)
_order_ = np.zeros( n, dtype=int )

sort_t = time.time()
i=0
for image_type in folder_image:
    for image in os.listdir(path_image + "/" + image_type):
        if i%5000 == 0: print "Sorting %dth data" %i
        _order_[i] = np.argmax( char_data[:,image_name_column] == '"'+image[0:-4]+'"' )
        i += 1   
print time.time() - sort_t

if sum( abs( np.arange(n)-np.sort(_order_) ) )==0: print 'sorting is 1-1'

feat_sort = feat_data[_order_,:]
char_sort = char_data[_order_,:]

## separate training / test data
np.random.seed(2016)
n_train = round(n*rate_train)
sample_train = np.random.choice(n, n_train, replace=False)
sample_train = np.sort(sample_train)

feat_train = feat_sort[sample_train,:]
feat_test = np.delete(feat_sort, sample_train, 0)
char_train = char_sort[sample_train,:]
char_test = np.delete(char_sort, sample_train, 0)


## shuffle training / test data
np.random.seed(1234)
shuffle_train = np.random.choice(feat_train.shape[0], feat_train.shape[0], replace=False)
shuffle_test = np.random.choice(feat_test.shape[0], feat_test.shape[0], replace=False)

feat_train = feat_train[shuffle_train,:]
feat_test = feat_test[shuffle_test,:]
char_train = char_train[shuffle_train,:]
char_test = char_test[shuffle_test,:]
print 'feat_train feat_test:', feat_train.shape, feat_test.shape
print 'char_train char_test:', char_train.shape, char_test.shape



###### save data
##outdata = file(out_data, "wb")
##np.savez(outdata, feat_train = feat_train, feat_test = feat_test)
##outdata.close()



#### export to excel
feat_data = np.load(out_data)
np.savetxt('L290A_train.csv', feat_train, delimiter=",")
np.savetxt('L290A_test.csv', feat_test, delimiter=",")
np.savetxt('L290Ach_train.csv', char_train, delimiter=",", fmt="%s")
np.savetxt('L290Ach_test.csv', char_test, delimiter=",", fmt="%s")
