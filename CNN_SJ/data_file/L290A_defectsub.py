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
path_image = "/home/nai/L290A_sub0405" # location of original images
resize_folder = "/home/nai/L290A_sub0405_96/"
char_train_name = "/home/nai/CNN_SJ/data_file/L290Ach_train.csv"
char_test_name  = "/home/nai/CNN_SJ/data_file/L290Ach_test.csv"
img_rows = 96
img_cols = 96

out_data = "L290A_defectsub.bin" # data file
nb_classes = 2
norm_tp = "ACSP1"
rate_train = 0.6

#### input - feature
feat_tr_valid_name = '/home/nai/CNN_SJ/data_file/L290A_train.csv'
feat_test_name = '/home/nai/CNN_SJ/data_file/L290A_test.csv'
#not_numeric_column = [0, 5, 6, 9]
#image_name_column = 2 # in not_numeric_column
out_feat = "L290A_defectsub_feature.bin"



#### load the information of original images
img_tr_valid_name = genfromtxt(char_train_name, skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
train_cut = round(  len( img_tr_valid_name )*0.8  )
img_train = img_tr_valid_name[ :train_cut,: ]
img_valid = img_tr_valid_name[ train_cut:,: ]

img_test  = genfromtxt(char_test_name,  skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
print 'img_train img_valid img_test', img_train.shape, img_valid.shape, img_test.shape



#### resize and save
folder_image = os.listdir(path_image)
X_train = np.zeros( (img_train.shape[0], img_rows, img_cols, 6) )
X_valid = np.zeros( (img_valid.shape[0], img_rows, img_cols, 6) )
X_test  = np.zeros( (img_test.shape[0],  img_rows, img_cols, 6) )
y_train, y_valid, y_test = [], [], []

i_train, i_valid, i_test = [0, 0, 0]
list_train, list_valid, list_test = [], [], []
for image_type in folder_image:
    print 'Resizing and saving', image_type, 'images'
    for image in os.listdir(path_image + "/" + image_type):
        image_name = '"'+image[0:-4].upper()+'"'
        if image_name in img_train[:,1]:
            list_train.append( image_name )
            img1 = Image.open(path_image + "/" + image_type + "/" + image)
            img1 = img1.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img1.save( resize_folder + image )
            img1 = Image.open( resize_folder + image )
            img1 = np.asarray( img1 )

            img2 = Image.open("/home/nai/L290A_defect0405/" + image_type + "/" + image)
            img2 = img2.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img2.save( "/home/nai/L290A_defect0405_96/" + image )
            img2 = Image.open( "/home/nai/L290A_defect0405_96/" + image )
            img2 = np.asarray( img2 )

            img = np.append(img1, img2, axis=2)

            X_train[i_train, :, :, :] = img
            y_train.append(  int( img_train[ list(img_train[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_train % 10000 ==0: print 'saving X_train', i_train
            i_train += 1
        elif image_name in img_valid[:,1]:
            list_valid.append( image_name )
            img1 = Image.open(path_image + "/" + image_type + "/" + image)
            img1 = img1.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img1.save( resize_folder + image )
            img1 = Image.open( resize_folder + image )
            img1 = np.asarray( img1 )

            img2 = Image.open("/home/nai/L290A_defect0405/" + image_type + "/" + image)
            img2 = img2.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img2.save( "/home/nai/L290A_defect0405_96/" + image )
            img2 = Image.open( "/home/nai/L290A_defect0405_96/" + image )
            img2 = np.asarray( img2 )

            img = np.append(img1, img2, axis=2)
            
            X_valid[i_valid, :, :, :] = img
            y_valid.append(  int( img_valid[ list(img_valid[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_valid % 10000 ==0: print 'saving X_valid', i_valid
            i_valid += 1
        elif image_name in img_test[:,1]:
            list_test.append( image_name )
            img1 = Image.open(path_image + "/" + image_type + "/" + image)
            img1 = img1.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img1.save( resize_folder + image )
            img1 = Image.open( resize_folder + image )
            img1 = np.asarray( img1 )

            img2 = Image.open("/home/nai/L290A_defect0405/" + image_type + "/" + image)
            img2 = img2.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img2.save( "/home/nai/L290A_defect0405_96/" + image )
            img2 = Image.open( "/home/nai/L290A_defect0405_96/" + image )
            img2 = np.asarray( img2 )

            img = np.append(img1, img2, axis=2)
            
            X_test[i_test, :, :, :] = img
            y_test.append(  int( img_test[ list(img_test[:,1]).index(image_name),0 ]=='"NG"' )  )
            if i_test % 10000 ==0:  print 'saving X_test' , i_test
            i_test +=1
            
print 'before X_train X_valid X_test', X_train.shape, X_valid.shape, X_test.shape

X_train = X_train[:i_train, :, :, :].transpose(0,3,1,2).astype('float32')/255
X_valid = X_valid[:i_valid, :, :, :].transpose(0,3,1,2).astype('float32')/255
X_test  = X_test[:i_test, :, :, :].transpose(0,3,1,2).astype('float32')/255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test  = np_utils.to_categorical(y_test,  nb_classes)

print 'X_train Y_train', X_train.shape, Y_train.shape
print 'Y_train 0, 1:', sum(Y_train[:,0]), sum(Y_train[:,1])
print 'X_valid Y_valid', X_valid.shape, Y_valid.shape
print 'Y_valid 0, 1:', sum(Y_valid[:,0]), sum(Y_valid[:,1])
print 'X_test Y_test', X_test.shape, Y_test.shape
print 'Y_test 0, 1:', sum(Y_test[:,0]), sum(Y_test[:,1])
print 'lists:', len(list_train), len(list_valid), len(list_test)

## save data
outdata = file(out_data, "wb")
np.savez(outdata, X_train = X_train, Y_train = Y_train,
         X_valid = X_valid, Y_valid = Y_valid,
         X_test  = X_test,  Y_test  = Y_test)
outdata.close()
