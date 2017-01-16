import sys
sys.path.append("C:\Users\NAI\.ipython")
import loaddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import cPickle as pickle


# get input file
input_file = raw_input("Enter the input file:") + '.csv' # C:\image_analysis\cap_code\cap_input
inputs = pd.read_csv(input_file)
inputs = inputs.set_index("input")

# get input
pdata = inputs.loc['python_data'].astype('str')["value"]
model_name = inputs.loc['model_name'].astype('str')["value"]

n_trees = inputs.loc['n_trees'].astype('int')["value"]
max_feature = inputs.loc['max_features'].astype('str')["value"]
criter = inputs.loc['criterion'].astype('str')["value"]

# load data
print 'loading', pdata + '.bin'
mydata = np.load(pdata  + '.bin')
X_train, X_test = mydata["X_train"], mydata["X_test"]
Y_train2, Y_test2 = mydata["Y_train2"], mydata["Y_test2"]
size1, size2 = X_train.shape[1], X_train.shape[2]
X_train = X_train.reshape(len(X_train), size1*size2*3)
X_test = X_test.reshape(len(X_test), size1*size2*3)

#model construction
print 'fitting', model_name
rf = RandomForestClassifier(n_estimators=n_trees, max_features = max_feature, criterion = criter)
rfmodel = rf.fit(X_train, Y_train2)
pred = rfmodel.predict(X_test)

print 'classification result'
table = confusion_matrix(Y_test2, pred)
print table

# save model
model_out = loaddata.rf_maeil(rfmodel, size1, size2)
with open( model_name + '.pkl', 'wb') as output:
    pickle.dump(model_out, output, -1)        
        
# save result
score = float(np.sum(np.diag(table))) / float(np.sum(table))
confidence = float(table[0, 0]) / float(np.sum(table[:, 0]))
detect1 = float(table[1, 1]) / float(np.sum(table[:, 1]))
detect2 = float(table[2, 2]) / float(np.sum(table[:, 2]))
detect3 = float(table[3, 3]) / float(np.sum(table[:, 3]))
detect4 = float(table[4, 4]) / float(np.sum(table[:, 4]))

result=['score', 'confidence', 'detect1', 'detect2', 'detect3', 'detect4', 'n_trees', 'max_feature', 'criterion']
result2=[score,confidence,detect1,detect2,detect3,detect4,n_trees, max_feature, criter]
result=np.vstack([result,result2])

#get_result
import csv
with open(model_name + '.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)

#  end
print 'generated %s and %s' % (model_name + '.csv', model_name + '.pkl')     