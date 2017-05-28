import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import os
import time
from tempfile import TemporaryFile

from PIL import Image

#### input - resize and load
path_image = "/home/nai/L290A_defect0405" # location of original images
resize_folder = "/home/nai/L290A_defect0405_160/"
char_train_name = "/home/nai/CNN_SJ/data_file/L290Ach_train.csv"
char_test_name  = "/home/nai/CNN_SJ/data_file/L290Ach_test.csv"
img_rows = 160
img_cols = 160

out_data = "L290A_defect0405.bin" # data file
nb_classes = 2
norm_tp = "ACSP1"
rate_train = 0.6

#### input - feature
feat_tr_valid_name = '/home/nai/CNN_SJ/data_file/L290A_train.csv'
feat_test_name = '/home/nai/CNN_SJ/data_file/L290A_test.csv'
#not_numeric_column = [0, 5, 6, 9]
#image_name_column = 2 # in not_numeric_column
out_feat = "L290A_defect0405_feature.bin"



#### load the information of original images
img_tr_valid_name = genfromtxt(char_train_name, skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
train_cut = round(  len( img_tr_valid_name )*0.8  )
img_train = img_tr_valid_name[ :train_cut,: ]
img_valid = img_tr_valid_name[ train_cut:,: ]

img_test  = genfromtxt(char_test_name,  skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
print 'img_train img_valid img_test', img_train.shape, img_valid.shape, img_test.shape



#### resize and save
folder_image = os.listdir(path_image)
X_train = np.zeros( (img_train.shape[0], img_rows, img_cols, 3) )
X_valid = np.zeros( (img_valid.shape[0], img_rows, img_cols, 3) )
X_test  = np.zeros( (img_test.shape[0],  img_rows, img_cols, 3) )
y_train, y_valid, y_test = [], [], []

i_train, i_valid, i_test = [0, 0, 0]
list_train, list_valid, list_test = [], [], []
for image_type in folder_image:
    print 'Resizing and saving', image_type, 'images'
    for image in os.listdir(path_image + "/" + image_type):
        image_name = '"'+image[0:-4].upper()+'"'
        if image_name in img_train[:,1]:
            list_train.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )

            X_train[i_train, :, :, :] = img
            y_train.append(  int( img_train[ list(img_train[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_train % 10000 ==0: print 'saving X_train', i_train
            i_train += 1
        elif image_name in img_valid[:,1]:
            list_valid.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )

            X_valid[i_valid, :, :, :] = img
            y_valid.append(  int( img_valid[ list(img_valid[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_valid % 10000 ==0: print 'saving X_valid', i_valid
            i_valid += 1
        elif image_name in img_test[:,1]:
            list_test.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )
            
            X_test[i_test, :, :, :] = img
            y_test.append(  int( img_test[ list(img_test[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_test % 10000 ==0:  print 'saving X_test' , i_test
            i_test +=1
##            
##print 'before X_train X_valid X_test', X_train.shape, X_valid.shape, X_test.shape
##
##X_train = X_train[:i_train, :, :, :].transpose(0,3,1,2).astype('float32')/255
##X_valid = X_valid[:i_valid, :, :, :].transpose(0,3,1,2).astype('float32')/255
##X_test  = X_test[:i_test, :, :, :].transpose(0,3,1,2).astype('float32')/255
##
##Y_train = np_utils.to_categorical(y_train, nb_classes)
##Y_valid = np_utils.to_categorical(y_valid, nb_classes)
##Y_test  = np_utils.to_categorical(y_test,  nb_classes)
##
##print 'X_train Y_train', X_train.shape, Y_train.shape
##print 'Y_train 0, 1:', sum(Y_train[:,0]), sum(Y_train[:,1])
##print 'X_valid Y_valid', X_valid.shape, Y_valid.shape
##print 'Y_valid 0, 1:', sum(Y_valid[:,0]), sum(Y_valid[:,1])
##print 'X_test Y_test', X_test.shape, Y_test.shape
##print 'Y_test 0, 1:', sum(Y_test[:,0]), sum(Y_test[:,1])
##print 'lists:', len(list_train), len(list_valid), len(list_test)
##
#### save data
##outdata = file(out_data, "wb")
##np.savez(outdata, X_train = X_train, Y_train = Y_train,
##         X_valid = X_valid, Y_valid = Y_valid,
##         X_test  = X_test,  Y_test  = Y_test)
##outdata.close()



#### load feature data in excel
index_train, index_valid, index_test = [], [], []
folder_image = os.listdir(path_image)
for image_type in folder_image:
    for image in os.listdir(path_image + "/" + image_type):
        image_name = '"'+image[0:-4].upper()+'"'
        if image_name in list_train: index_train.append(  list(img_train[:,1]).index(image_name)  )
        elif image_name in list_valid: index_valid.append(  list(img_valid[:,1]).index(image_name)  )
        elif image_name in list_test: index_test.append(  list(img_test[:,1]).index(image_name)  )

index_train = np.sort( np.asarray(index_train) )
index_valid = np.sort( np.asarray(index_valid) )
index_test  = np.sort( np.asarray(index_test)  )

feat_tr_valid = genfromtxt(feat_tr_valid_name, skip_header=0, dtype='float32', delimiter=',')
feat_test     = genfromtxt(feat_test_name,     skip_header=0, dtype='float32', delimiter=',')
train_cut = round(  feat_tr_valid.shape[0]*0.8  )
feat_train = feat_tr_valid[ :train_cut,: ]
feat_valid = feat_tr_valid[ train_cut:,: ]

feat_train = feat_train[ index_train,: ]
feat_valid = feat_valid[ index_valid,: ]
feat_test  = feat_test[  index_test, : ]
print 'feat_train feat_valid feat_test', feat_train.shape, feat_valid.shape, feat_test.shape

## save feature data
outdata = file(out_feat, "wb")
np.savez(outdata, feat_train = feat_train, feat_valid = feat_valid, feat_test = feat_test)
outdata.close()
