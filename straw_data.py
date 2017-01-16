import sys
sys.path.append("C:\Users\NAI\.ipython")
from loaddata import img_to_array
import loaddata
import numpy as np
import random
from keras.utils import np_utils

# get input file
get_input = loaddata.load_data_input()
data_dir = get_input.data_dir # C:\image_analysis\straw_data
out_data = get_input.out_data + '.bin' # C:\image_analysis\straw_data\strawdata
size1 = get_input.size1
size2 = get_input.size2

# get array data    
straw = img_to_array(data_dir, 'straw', size1, size2)
strawexist = img_to_array(data_dir, 'strawexist', size1, size2)
strawlocation = img_to_array(data_dir, 'strawlocation', size1, size2)  
    
# straw data for cnn (merge 3 type arrays )
img_final = np.vstack([straw, strawexist, strawlocation])
img_final = img_final.transpose(0,3,1,2).astype('float32')/255

# last processing
#creat y: straw2 is 0 (normal), strawexist2 is 1 (abnormal1), strawlocation2 is 2 (abnormal2)
a=np.repeat(0, [len(straw)])
b=np.repeat(1, [len(strawexist)])
c=np.repeat(2, [len(strawlocation)])

#concatenate y
d=np.concatenate([a,b,c])
e=d

#convert 0,1,2 to (1,0,0), (0,1,0), (0,0,1) for deep learning
d = np_utils.to_categorical(d,3)
length = len(d)

#rearrange the image numbers
random.seed(1)
idx=random.sample(range(length), length/4) #idx is indexs for test set. the proportion is 1 over 4
idx_IN_rows = [i for i in range(length) if i not in idx] #idx_IN_rows is index for training set. the proportion is 3 over 4

#divide traning and test set for each x and y
X_train=img_final[idx_IN_rows]
X_test=img_final[idx]
Y_train=d[idx_IN_rows]
Y_train2=e[idx_IN_rows] #put two different types of Y_train. this is needed in deep learning
Y_test2=e[idx]

# save out_data
outdata = file(out_data, "wb")
np.savez(outdata, X_train = X_train, Y_train = Y_train, Y_train2 = Y_train2,
         X_test = X_test,  Y_test2 = Y_test2)
outdata.close()
