import sys
sys.path.append("C:\Users\NAI\.ipython")
from loaddata import img_to_array
import loaddata
import numpy as np
import cPickle as pickle
import glob, csv
from keras.models import model_from_yaml

# class for cnn model(has size and cutoff information)
cnn_maeil = loaddata.cnn_maeil        
        
# get input
if True:
    model_name = raw_input("Enter the model: ") # C:\image_analysis\straw_result\straw_cnn
    data_dir = raw_input("Enter the image data folder: ") # C:\image_analysis\straw_data
    result_name = raw_input("Set the result file: ") # C:\image_analysis\straw_result\newresult
else:
    model_name = 'C:\image_analysis\straw_result\straw_cnn'
    data_dir = 'C:\image_analysis\straw_data'
    result_name = 'C:\image_analysis\straw_result\newresult' 

# load model
print 'loading model %s' % model_name
with open(model_name + '.pkl', 'rb') as input:
    model_input = pickle.load(input)    

model = model_from_yaml(model_input.model_yaml)
model.load_weights( model_name + '.h5' )
size1, size2, cutoff = model_input.nrow, model_input.ncol, model_input.cutoff

# load, resize and get array from image data
print 'loading images from %s\\newdata' % data_dir
img_final = img_to_array(data_dir, 'newdata', size1, size2).transpose(0, 3, 1, 2).astype('float32')/255

# get pred_class from pred_prob
pred_prob = model.predict_proba(img_final, 20)
pred_class = np.zeros(len(pred_prob))
pred_class[pred_prob[:, 0] <= cutoff] = np.argmax(pred_prob[pred_prob[:, 0] <= cutoff, 1:], axis = 1) + 1

# save result
data_list = glob.glob(data_dir + "\\newdata\*")
result = np.vstack([data_list, pred_class]).transpose(1,0)

with open(result_name + '.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)
