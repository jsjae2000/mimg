import sys
sys.path.append("C:\Users\NAI\.ipython")
from loaddata import img_to_array
import loaddata
import numpy as np
import cPickle as pickle
import glob, csv

# class for cnn model(has size and cutoff information)
rf_maeil = loaddata.rf_maeil    

# get input
if True:
    model_name = raw_input("Enter the model: ") # C:\image_analysis\cap_result\cap_rf
    data_dir = raw_input("Enter the image data folder: ") # C:\image_analysis\cap_data
    result_name = raw_input("Set the result file: ") # C:\image_analysis\cap_result\newresult
else:
    model_name = 'C:\image_analysis\cap_result\cap_rf'
    data_dir = 'C:\image_analysis\cap_data'
    result_name = 'C:\image_analysis\cap_result\\newresult' 

# load model
print 'loading model %s' % model_name
with open(model_name + '.pkl', 'rb') as input:
    model_input = pickle.load(input)    
model, size1, size2 = model_input.model, model_input.nrow, model_input.ncol

# load, resize and get array from image data
print 'loading images from %s\\newdata' % data_dir
img_final = img_to_array(data_dir, 'newdata', size1, size2)
img_final = img_final.reshape(len(img_final), size1*size2*3).astype('float32')/255

# get result
pred_class = model.predict(img_final)

# save result
data_list = glob.glob(data_dir + "\\newdata\*")
result = np.vstack([data_list, pred_class]).transpose(1,0)

with open(result_name + '.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)
