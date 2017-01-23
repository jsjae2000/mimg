import glob, PIL
import numpy as np
from PIL import Image

class load_data_input:
    def __init__(self):
        self.data_dir = raw_input("Enter the image data folder: ") # C:\image_analysis\cap_data
        self.out_data = raw_input("Set the name of python data: ") # C:\image_analysis\cap_data\capdata
        self.size1 = int(raw_input("Set the row size of images you want to use: "))
        self.size2 = int(raw_input("Set the column size of images you want to use : "))

def img_to_array(data_dir, img_type, size1, size2):
    data_list=glob.glob(data_dir + "\%s\*" % img_type)
    img_array = np.zeros( (len(data_list), size1, size2, 3) )
    for k in range(len(data_list)):
        img = Image.open(data_list[k])
        img = img.resize((size2,size1), PIL.Image.ANTIALIAS)
        img_array[k, :, :, :] = np.asarray(img)
    return(img_array) 
    
class cnn_maeil:
    def __init__(self, model_yaml, size1, size2, cutoff):
        self.model_yaml = model_yaml
        self.nrow = size1
        self.ncol = size2
        self.cutoff = cutoff    
    
class rf_maeil:
    def __init__(self, model, size1, size2):
        self.model = model
        self.nrow = size1
        self.ncol = size2