#cap data
import numpy as np
import glob, PIL, os, random
from PIL import Image
from time import localtime, strftime
from keras.utils import np_utils

# get input file
data_dir = raw_input("Enter the image data folder: ")
out_data = raw_input("Enter the python data(.bin): ")
size1 = int(raw_input("Enter the row size of images you want to use for model: "))
size2 = int(raw_input("Enter the column size of images you want to use for model: "))

# load, resize image data
os.chdir(data_dir)

# capdata
def img_dt(data_dir, img_type):
    resize_dir = '%s_tmp' % img_type + strftime('%Y%m%d%H%M%S', localtime())
    os.mkdir(resize_dir)
    
    data_list=glob.glob(data_dir + "\%s\*" % img_type)
    img_array = np.zeros( (len(data_list), size1, size2, 3) )
    for k in range(len(data_list)):
        rimg_name = resize_dir + data_list[k].split('\\')[-1]
        img = Image.open(data_list[k])
        img = img.resize((size2,size1), PIL.Image.ANTIALIAS)
        img.save(rimg_name)
        img = Image.open( rimg_name )
        img = np.asarray( img )
        img_array[k, :, :, :] = img
        os.remove(rimg_name)
    os.rmdir(resize_dir)
    return(img_array)