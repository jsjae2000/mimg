import sys
sys.path.append("C:\Users\NAI\.ipython")
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import cPickle as pickle
import math
import loaddata

# get input file
input_file = raw_input("Enter the input file:") + '.csv' # C:\image_analysis\straw_code\straw_input
inputs = pd.read_csv(input_file)
inputs = inputs.set_index("input")

# get inputs from input file
pdata = inputs.loc['python_data'].astype('str')["value"]
model_name = inputs.loc['model_name'].astype('str')["value"]

batch_size = inputs.loc['batchsize'].astype('int')["value"]
nb_epoch = inputs.loc['nb_epoch'].astype('int')["value"]
filter_nb = inputs.loc['nb_filter'].astype('int')["value"]
dense_nb = inputs.loc['nb_dense'].astype('int')["value"]
activation = inputs.loc['activation'].astype('str')["value"]
initial = inputs.loc['initial'].astype('str')["value"]
optimizer = inputs.loc['optimizer'].astype('str')["value"]

# set default inputs
#tuning parts
##batch_size=20
##nb_epoch=15
patience=3#
grid=100#
confidence_level=0.99#
pool_size=2#
monitor='val_loss'#
##filter_nb=16
filter_size=3#
##dense_nb=48
validation_split = 0.1#
#conv_drop_out=0.5
#dense_drop_out=0.3
#activation2=['softplus','softsign','relu','tanh','sigmoid']
#initial2=['uniform','lecun_uniform','normal','zero','glorot_normal','glorot_uniform','he_normal','he_uniform']
#loss2=['poisson','cosine_proximity','squared_hinge','hinge','categorical_crossentropy']
loss='categorical_crossentropy'
#optimizer2=['sgd','rmsprop','adadelta']

# load data
print 'loading', pdata + '.bin'
mydata = np.load( pdata + '.bin' )
X_train, X_test = mydata["X_train"], mydata["X_test"]
Y_train, Y_train2, Y_test2 = mydata["Y_train"], mydata["Y_train2"], mydata["Y_test2"]

size1, size2 = X_train.shape[2:4]
nb_classes = Y_train.shape[1]

train_length=len(Y_train)
result=['score','confidence','detect1','detect2','nb_filter_nb','nb_dense','activation','initial','optimizer','cutoff']

X_val=X_train[range(int(train_length*(1-validation_split)),train_length)]
Y_val=Y_train[range(int(train_length*(1-validation_split)),train_length)]
Y_val2=Y_train2[range(int(train_length*(1-validation_split)),train_length)]

#result=['table','score','confidence','detect1','detect2','cutoff','filter_nb','filter_size','dense_nb','conv_drop_out','dense_drop_out','pool_size','activation','initial','loss','optimizer','monitor','validation_split']

#model construction
print 'fitting', model_name
model = Sequential()
model.add(Convolution2D(filter_nb, filter_size, filter_size, border_mode='valid', input_shape=(3, size1, size2), activation=activation))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
model.add(Dropout(0.5))
#model.add(Convolution2D(filter_nb, filter_size, filter_size, border_mode='valid', input_shape=(3, size1, size2), activation=activation))
#model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))
#model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(dense_nb,init=initial,activation=activation))
#model.add(Dropout(0.5))
#model.add(Dense(dense_nb,init=initial,activation=activation))
#model.add(Dropout(0.5))
#model.add(Dense(dense_nb,init=initial,activation=activation))
#model.add(Dropout(0.5))
model.add(Dense(dense_nb,init=initial,activation=activation))
model.add(Dropout(0.5))
model.add(Dense(dense_nb,init=initial,activation=activation))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,init=initial))
model.add(Activation('softmax'))
#model compile
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#model fit
ES=EarlyStopping(monitor=monitor,patience=patience,verbose=1)
check=ModelCheckpoint(filepath=model_name+'.h5',monitor=monitor,verbose=1,save_best_only=True)
model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_split=validation_split,callbacks=[ES, check])

#validation
#score =model.evaluate(X_val, Y_val, batch_size=batch_size)
clas=model.predict_classes(X_val, batch_size=batch_size)
proba=model.predict_proba(X_val, batch_size=batch_size)
#get cutoff
real0=len(Y_val2[Y_val2==0])
real1=len(Y_val2)-real0
select=np.repeat(0.,[grid])
location=np.repeat(0.,[grid])
range_start=int(math.ceil(grid/float(3)))
for i in range(range_start,grid-1):
    gg=0.0
    bb=0.0
    for j in range(len(clas)):
        if(float(proba[j][0])<(float(i)/grid)):
            if(float(Y_val2[j])>0.5):
                bb=bb+1.0
        if(float(proba[j][0])>=(float(i)/grid)):
            if(float(Y_val2[j])<0.5):
                gg=gg+1.0
    gb=real1-bb #good predict true bad
    if(float(gg)>confidence_level*float(gg+gb)):
        select[i]=1
        location[i]=i
if(np.sum(select)<1):
    cutoff=0.9999
if(np.sum(select)>0):
    cutoff = min(location[np.where(select==1)]/float(grid))
#test
clas=model.predict_classes(X_test, batch_size=batch_size)
proba=model.predict_proba(X_test, batch_size=batch_size)

# sj test
pred_class = np.zeros(len(proba))
pred_class[proba[:, 0] <= cutoff] = np.argmax(proba[proba[:, 0] <= cutoff, 1:], axis = 1) + 1
print 'classification result'
table = confusion_matrix(Y_test2, pred_class)
print table

# get confidence and detect
score = float(np.sum(np.diag(table))) / float(np.sum(table))
if float(np.sum(table[:, 0])) == 0: 
    confidence = 0
else: confidence = float(table[0, 0]) / float(np.sum(table[:, 0]))
    
if float(np.sum(table[:, 1])) == 0: detect1 = 0
else: detect1 = float(table[1, 1]) / float(np.sum(table[:, 1]))
    
if float(np.sum(table[:, 2])) == 0: detect2 = 0
else: detect2 = float(table[2, 2]) / float(np.sum(table[:, 2]))
    
if float(np.sum(table[:, 0])) == 0:
    confidence=0
    detect1=0
    detect2=0

print('score',score)                                                    
print('confidence',confidence)
print('detect1',detect1)
print('detect2',detect2)
result2=[score,confidence,detect1,detect2,filter_nb,dense_nb,activation,initial,optimizer,cutoff]
result=np.vstack([result,result2])

#get_result
import csv
with open(model_name + '.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)

# save model : yaml, size, cutoff
model_out = loaddata.cnn_maeil(model.to_yaml(), size1, size2, cutoff)    
with open( model_name + '.pkl', 'wb') as output:
    pickle.dump(model_out, output, -1)

#  end
print 'generating %s , %s and %s' % (model_name + '.csv', model_name + '.pkl', model_name + '.h5' )
