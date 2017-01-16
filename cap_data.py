import sys
sys.path.append("C:\Users\NAI\.ipython")
from loaddata import img_to_array
import loaddata
import numpy as np
import random
from keras.utils import np_utils

# get input file
get_input = loaddata.load_data_input()
data_dir = get_input.data_dir # C:\image_analysis\cap_data
out_data = get_input.out_data + '.bin' # C:\image_analysis\cap_data\capdata
size1 = get_input.size1
size2 = get_input.size2

# get array data       
cap = img_to_array(data_dir, 'cap', size1, size2)
capexist = img_to_array(data_dir, 'capexist', size1, size2)
capbestbefore = img_to_array(data_dir, 'capbestbefore', size1, size2)  
capbestbeforecut = img_to_array(data_dir, 'capbestbeforecut', size1, size2)
capbestbeforecutup = img_to_array(data_dir, 'capbestbeforecutup', size1, size2)

# cap data make for rf
img_final = np.vstack([cap, capexist, capbestbefore, capbestbeforecut, capbestbeforecutup])
img_final = img_final.astype('float32')/255

# last processing                    
a=np.repeat(0, [len(cap)])
b=np.repeat(1, [len(capexist)])
c=np.repeat(2, [len(capbestbefore)])
d=np.repeat(3, [len(capbestbeforecut)])
d2=np.repeat(4, [len(capbestbeforecutup)])
e=np.concatenate([a,b,c,d,d2])
f=e

e = np_utils.to_categorical(e,5)
length = len(e)
random.seed(1)
idx=random.sample(range(length), length/4)
idx_IN_rows = [i for i in range(length) if i not in idx]

X_train=img_final[idx_IN_rows]
X_test=img_final[idx]
Y_train2=f[idx_IN_rows]
Y_test2=f[idx]

# save out_data
outdata = file(out_data, "wb")
np.savez(outdata, X_train = X_train, Y_train2 = Y_train2,
         X_test = X_test,  Y_test2 = Y_test2)
outdata.close()