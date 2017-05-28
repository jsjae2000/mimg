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

#### global input
_model_ = "L290A_part"
img_size = 224

#### input - resize and load
path_image = "/home/nai/sL290A" # location of original images
resize_folder = "/home/nai/sL290A_" + str(img_size) + "/"
char_train_name = "/home/nai/CNN_SJ/data_file/L290Ach_train.csv"
char_test_name  = "/home/nai/CNN_SJ/data_file/L290Ach_test.csv"
img_rows, img_cols = img_size, img_size

out_data = _model_ + ".bin" # data file
nb_classes = 2
norm_tp = "SDSP1"
rate_train = 0.48
rate_valid = 0.12

#### input - feature
feat_tr_valid_name = '/home/nai/CNN_SJ/data_file/L290A_train.csv'
feat_test_name = '/home/nai/CNN_SJ/data_file/L290A_test.csv'
#not_numeric_column = [0, 5, 6, 9]
#image_name_column = 2 # in not_numeric_column
out_feat = _model_ + "_feature.bin"



#### n?
n=0
folder_image = os.listdir(path_image)
for image_type in folder_image:
    n += len( os.listdir(path_image + "/" + image_type) )
print "the number of total images:", n



#### load the information of original images
img_tr_valid_name = genfromtxt(char_train_name, skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
train_cut = round(  len( img_tr_valid_name )*0.8  )
img_train = img_tr_valid_name[ :train_cut,: ]
img_valid = img_tr_valid_name[ train_cut:,: ]

img_test  = genfromtxt(char_test_name,  skip_header=0, dtype=np.str, delimiter=',')[:,[0,2]]
print 'img_train img_valid img_test', img_train.shape, img_valid.shape, img_test.shape

part_tr_valid = genfromtxt(char_train_name, skip_header=0, dtype=np.str, delimiter=',')[:,[1,3]]
part_train = part_tr_valid[ :train_cut ]
part_valid = part_tr_valid[ train_cut: ]
part_test = genfromtxt(char_test_name, skip_header=0, dtype=np.str, delimiter=',')[:,[1,3]]
print 'part_train part_valid part_test', part_train.shape, part_valid.shape, part_test.shape



#### resize and save
folder_image = os.listdir(path_image)
X_train = np.zeros( (img_train.shape[0], img_rows, img_cols, 3) )
X_valid = np.zeros( (img_valid.shape[0], img_rows, img_cols, 3) )
X_test  = np.zeros( (img_test.shape[0],  img_rows, img_cols, 3) )
y_train, y_valid, y_test = [], [], []
aoi_train, aoi_valid, aoi_test = [], [], []

i_train, i_valid, i_test = [0, 0, 0]
list_train, list_valid, list_test = [], [], []
for image_type in folder_image:
    print 'Resizing and saving', image_type, 'images'
    for image in os.listdir(path_image + "/" + image_type):
        image_name = '"'+image[0:-4].upper()+'"'
        if image_name in img_train[:,1]:
          if part_train[ list(img_train[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            list_train.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )

            X_train[i_train, :, :, :] = img
            y_train.append(  int( img_train[ list(img_train[:,1]).index(image_name),0 ]=='"NG"' )  )
            aoi_train.append(  int( part_train[ list(img_train[:,1]).index(image_name),0 ][7] )  )
            #if i_train % 10000 ==0: print 'saving X_train', i_train
            i_train += 1
        elif image_name in img_valid[:,1]:
          if part_valid[ list(img_valid[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            list_valid.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )

            X_valid[i_valid, :, :, :] = img
            y_valid.append(  int( img_valid[ list(img_valid[:,1]).index(image_name),0 ]=='"NG"' )  )
            aoi_valid.append(  int( part_valid[ list(img_valid[:,1]).index(image_name),0 ][7] )  )
            #if i_valid % 10000 ==0: print 'saving X_valid', i_valid
            i_valid += 1
        elif image_name in img_test[:,1]:
          if part_test[ list(img_test[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            list_test.append( image_name )
            img = Image.open(path_image + "/" + image_type + "/" + image)
            img = img.resize( (img_cols, img_rows), Image.ANTIALIAS )
            img.save( resize_folder + image )
            img = Image.open( resize_folder + image )
            img = np.asarray( img )
            
            X_test[i_test, :, :, :] = img
            y_test.append(  int( img_test[ list(img_test[:,1]).index(image_name),0 ]=='"NG"' )  )
            aoi_test.append(  int( part_test[ list(img_test[:,1]).index(image_name),0 ][7] )  )
            #if i_test % 10000 ==0:  print 'saving X_test' , i_test
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
##
#### save data - tmp
##outdata = file('asdf.bin', "wb")
##np.savez(outdata, X_train = X_train, Y_train = Y_train,
##         X_valid = X_valid, Y_valid = Y_valid,
##         X_test  = X_test,  Y_test  = Y_test,
##         a = aoi_train, b = aoi_valid, c = aoi_test)
##outdata.close()

##mydata = np.load( 'asdf.bin' )
##X_train = mydata["X_train"]
##Y_train = mydata["Y_train"]
##X_valid = mydata["X_valid"]
##Y_valid = mydata["Y_valid"]
##X_test  = mydata["X_test"]
##Y_test  = mydata["Y_test"]
##aoi_train = mydata["a"]
##aoi_valid = mydata["b"]
##aoi_test =  mydata["c"]

#### divide part
aoi_train, aoi_valid, aoi_test = np.asarray(aoi_train), np.asarray(aoi_valid), np.asarray(aoi_test)
X_train2, Y_train2 = X_train[ np.where(aoi_train==2)[0], : ,: ,:], Y_train[ np.where(aoi_train==2)[0], :]
X_valid2, Y_valid2 = X_valid[ np.where(aoi_valid==2)[0], : ,: ,:], Y_valid[ np.where(aoi_valid==2)[0], :]
X_test2, Y_test2   = X_test[  np.where(aoi_test==2)[0], : ,: , :], Y_test[  np.where(aoi_test==2)[0],  :]

X_train4, Y_train4 = X_train[ np.where(aoi_train==4)[0], : ,: ,:], Y_train[ np.where(aoi_train==4)[0], :]
X_valid4, Y_valid4 = X_valid[ np.where(aoi_valid==4)[0], : ,: ,:], Y_valid[ np.where(aoi_valid==4)[0], :]
X_test4, Y_test4   = X_test[  np.where(aoi_test==4)[0], : ,: , :], Y_test[  np.where(aoi_test==4)[0],  :]

X_train6, Y_train6 = X_train[ np.where(aoi_train==6)[0], : ,: ,:], Y_train[ np.where(aoi_train==6)[0], :]
X_valid6, Y_valid6 = X_valid[ np.where(aoi_valid==6)[0], : ,: ,:], Y_valid[ np.where(aoi_valid==6)[0], :]
X_test6, Y_test6   = X_test[  np.where(aoi_test==6)[0], : ,: , :], Y_test[  np.where(aoi_test==6)[0],  :]

X_train7, Y_train7 = X_train[ np.where(aoi_train==7)[0], : ,: ,:], Y_train[ np.where(aoi_train==7)[0], :]
X_valid7, Y_valid7 = X_valid[ np.where(aoi_valid==7)[0], : ,: ,:], Y_valid[ np.where(aoi_valid==7)[0], :]
X_test7, Y_test7   = X_test[  np.where(aoi_test==7)[0], : ,: , :], Y_test[  np.where(aoi_test==7)[0],  :]

X_train9, Y_train9 = X_train[ np.where(aoi_train==9)[0], : ,: ,:], Y_train[ np.where(aoi_train==9)[0], :]
X_valid9, Y_valid9 = X_valid[ np.where(aoi_valid==9)[0], : ,: ,:], Y_valid[ np.where(aoi_valid==9)[0], :]
X_test9, Y_test9   = X_test[  np.where(aoi_test==9)[0], : ,: , :], Y_test[  np.where(aoi_test==9)[0],  :]

print 'X_train2 X_valid2 X_test2', X_train2.shape, X_valid2.shape, X_test2.shape
print 'X_train4 X_valid4 X_test4', X_train4.shape, X_valid4.shape, X_test4.shape
print 'X_train6 X_valid6 X_test6', X_train6.shape, X_valid6.shape, X_test6.shape
print 'X_train7 X_valid7 X_test7', X_train7.shape, X_valid7.shape, X_test7.shape
print 'X_train9 X_valid9 X_test9', X_train9.shape, X_valid9.shape, X_test9.shape

## save data
outdata = file(out_data, "wb")
np.savez(outdata, X_train2 = X_train2, Y_train2 = Y_train2,
         X_valid2 = X_valid2, Y_valid2 = Y_valid2,
         X_test2  = X_test2,  Y_test2  = Y_test2,
         X_train4 = X_train4, Y_train4 = Y_train4,
         X_valid4 = X_valid4, Y_valid4 = Y_valid4,
         X_test4  = X_test4,  Y_test4  = Y_test4,
         X_train6 = X_train6, Y_train6 = Y_train6,
         X_valid6 = X_valid6, Y_valid6 = Y_valid6,
         X_test6  = X_test6,  Y_test6  = Y_test6,
         X_train7 = X_train7, Y_train7 = Y_train7,
         X_valid7 = X_valid7, Y_valid7 = Y_valid7,
         X_test7  = X_test7,  Y_test7  = Y_test7,
         X_train9 = X_train9, Y_train9 = Y_train9,
         X_valid9 = X_valid9, Y_valid9 = Y_valid9,
         X_test9  = X_test9,  Y_test9  = Y_test9,
         )
outdata.close()



#### load feature data in excel
index_train, index_valid, index_test = [], [], []
folder_image = os.listdir(path_image)
for image_type in folder_image:
    for image in os.listdir(path_image + "/" + image_type):
        image_name = '"'+image[0:-4].upper()+'"'
        if image_name in list_train:
          if part_train[ list(img_train[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            index_train.append(  list(img_train[:,1]).index(image_name)  )
        elif image_name in list_valid:
          if part_valid[ list(img_valid[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            index_valid.append(  list(img_valid[:,1]).index(image_name)  )
        elif image_name in list_test:
          if part_test[ list(img_test[:,1]).index(image_name),1 ] == '"B6A055FH5LA03"':
            index_test.append(  list(img_test[:,1]).index(image_name)  )

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

feat_train2 = feat_train[ np.where(aoi_train==2)[0], :]
feat_valid2 = feat_valid[ np.where(aoi_valid==2)[0], :]
feat_test2  = feat_test[  np.where(aoi_test==2)[0],  :]

feat_train4 = feat_train[ np.where(aoi_train==4)[0], :]
feat_valid4 = feat_valid[ np.where(aoi_valid==4)[0], :]
feat_test4  = feat_test[  np.where(aoi_test==4)[0],  :]

feat_train6 = feat_train[ np.where(aoi_train==6)[0], :]
feat_valid6 = feat_valid[ np.where(aoi_valid==6)[0], :]
feat_test6  = feat_test[  np.where(aoi_test==6)[0],  :]

feat_train7 = feat_train[ np.where(aoi_train==7)[0], :]
feat_valid7 = feat_valid[ np.where(aoi_valid==7)[0], :]
feat_test7  = feat_test[  np.where(aoi_test==7)[0],  :]

feat_train9 = feat_train[ np.where(aoi_train==9)[0], :]
feat_valid9 = feat_valid[ np.where(aoi_valid==9)[0], :]
feat_test9  = feat_test[  np.where(aoi_test==9)[0],  :]

## save feature data
outdata = file(out_feat, "wb")
np.savez(outdata, feat_train2 = feat_train2, feat_valid2 = feat_valid2, feat_test2 = feat_test2,
         feat_train4 = feat_train4, feat_valid4 = feat_valid4, feat_test4 = feat_test4,
         feat_train6 = feat_train6, feat_valid6 = feat_valid6, feat_test6 = feat_test6,
         feat_train7 = feat_train7, feat_valid7 = feat_valid7, feat_test7 = feat_test7,
         feat_train9 = feat_train9, feat_valid9 = feat_valid9, feat_test9 = feat_test9
         )
outdata.close()


