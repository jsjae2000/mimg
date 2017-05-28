'''
GPU run command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sL691A_v07.py
'''
import numpy as np
import keras
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import theano
import yaml
import time

np.random.seed(1337) # for reproducibility
print '\n'

#### input
## input - common
_model_ = 'L290A_part2'
_mydata_ = '/home/nai/CNN_SJ/data_file/L290A_part.bin'

## input - model
batch_size = 24
nb_epoch = 5000

## input - activation
get_layer_number_in = [31]
batch_size_act = batch_size

## input - probability
batch_size_prob = batch_size

## input - search optimal cutoff
len_search = 2000 # length.out in (0,1)
conf_level_in = [0.98] # confidence level
batch_size_tun = batch_size # batch_size
print_rate = True


#### load data
mydata = np.load( _mydata_ )
X_train, X_valid = mydata["X_train2"], mydata["X_valid2"]
Y_train, Y_valid = mydata["Y_train2"], mydata["Y_valid2"]
mean_train = np.mean(X_train, axis=0)
X_train -= mean_train
X_valid -= mean_train
#del(mydata)
#del(mean_train)

#del( mydata )
print 'load X_train X_valid:', X_train.shape, X_valid.shape
print 'load Y_train Y_valid:', Y_train.shape, Y_valid.shape,'\n'
img_rows, img_cols = X_train.shape[2:4]
nb_classes = Y_train.shape[1]

#### model construction
#model.add(Activation('relu'))
#model.add(PReLUz())

frac = 16
frac2 = 1
model = Sequential()

model.add( Convolution2D(64/frac*frac2, 3, 3, border_mode='same', input_shape=(3, img_rows, img_cols)) )
model.add( Activation('relu') )
model.add( Convolution2D(64/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add( Convolution2D(128/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(128/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add( Convolution2D(256/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(256/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(256/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( Convolution2D(512/frac*frac2, 3, 3, border_mode='same') )
model.add( Activation('relu') )
model.add( MaxPooling2D( pool_size=(2, 2) ) )

model.add( Flatten() )
model.add( Dense(4096/frac*frac2) )
model.add( Activation('relu') )
model.add( Dropout(0.5) )
model.add( Dense(4096/frac*frac2) )
model.add( Activation('relu') )
model.add( Dropout(0.5) )
model.add( Dense(nb_classes) )
model.add( Activation('softmax') )

##sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rms = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-06)
model.compile(loss='binary_crossentropy', optimizer=rms)

hist = model.fit(X_train, Y_train, batch_size=batch_size, 
                 nb_epoch=nb_epoch, show_accuracy=True, 
                 verbose=1, validation_data=(X_valid, Y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                            ModelCheckpoint(filepath=_model_+'.h5', monitor='val_loss', save_best_only=True)]
                 )
                 
## save the model
model_yaml = model.to_yaml(); del(model)
open( _model_+'.yaml', 'w' ).write(model_yaml)
print 'save', _model_, 'at', _model_+'.yaml'

## save History
with open( _model_+'_history.yaml','w' ) as fp:
    yaml.dump(hist.history, fp)
print 'save', _model_, 'history at', _model_+'_history.yaml \n'



#### load model - use at activation & probability & cutoff
model = model_from_yaml(open( _model_+'.yaml' ).read())
model.load_weights( _model_+'.h5' )
print 'load', _model_, '\n'

## test
X_test = mydata["X_test2"]
Y_test = mydata["Y_test2"]
X_test  -= mean_train
print 'load Y_valid, Y_test:', Y_valid.shape, Y_test.shape,'\n'
#img_rows, img_cols = X_valid.shape[2:4]
#nb_classes = Y_valid.shape[1]

score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=1)
print 'Valid score Valid accuracy:', score[0], score[1], '\n'



###### get and save activation
#### get activation
##def _get_layer_(model, layer, X_data, batch_size_act):
##    get_layer = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False))
##    seq_batch = range(0,X_data.shape[0],batch_size_act) + [X_data.shape[0]]
##
##    for i in range(1, len(seq_batch)):
##        if (i*batch_size_act)%10000==0: print 'get layer at batch:', seq_batch[i],'/',X_data.shape[0]
##        X_batch = X_data[ seq_batch[i-1]:seq_batch[i],:,:,: ]
##        if i==1: result = get_layer(X_batch)
##        else: result = np.append(result, get_layer(X_batch), axis=0)
##    return result
##
##for get_layer_number in get_layer_number_in:
##    act_train = _get_layer_(model, get_layer_number, X_train, batch_size_act)
##    #act_train = act_train.reshape( act_train.shape[0],act_train.shape[1]*act_train.shape[2]*act_train.shape[3] )
##    print 'get act_train:', act_train.shape
##    act_valid = _get_layer_(model, get_layer_number, X_valid, batch_size_act)
##    #act_valid = act_valid.reshape( act_valid.shape[0],act_valid.shape[1]*act_valid.shape[2]*act_valid.shape[3] )
##    print 'get act_valid:', act_valid.shape
##    act_test  = _get_layer_(model, get_layer_number, X_test,  batch_size_act)
##    #act_test = act_test.reshape( act_test.shape[0],act_test.shape[1]*act_test.shape[2]*act_test.shape[3] )
##    print 'get act_test:',  act_test.shape
##
##    ## save act data
##    outdata = file( _model_ + '_act' + str(get_layer_number) + '.bin', "wb" )
##    np.savez(outdata, act_train=act_train, act_valid=act_valid, act_test=act_test, Y_train=Y_train, Y_valid=Y_valid, Y_test=Y_test)
##    outdata.close()
##    print 'save act_train act_valid act_test Y_train Y_valid Y_test:', act_train.shape, act_valid.shape, act_test.shape, Y_train.shape, Y_valid.shape, Y_test.shape
##    print 'at ' + _model_ + '_act' + str(get_layer_number) + '.bin \n'



#### get and save probability
## get probability
print('get probability')
prob_valid = model.predict_proba(X_valid, batch_size_prob)
print 'get prob_valid:', prob_valid.shape
prob_test  = model.predict_proba(X_test,  batch_size_prob)
print 'get prob_test:', prob_test.shape
##
##    ## save probability
##    outdata = file( _model_+'_prob.bin', "wb") ####
##    np.savez(outdata, prob_valid = prob_valid, prob_test = prob_test, Y_valid = Y_valid, Y_test = Y_test)
##    outdata.close()
##    print 'save prob_valid prob_test Y_valid Y_test:', prob_valid.shape, prob_test.shape, Y_valid.shape, Y_test.shape
##    print 'at ' + _model_ + '_prob.bin \n'
##
##
##
##    #### search optimal cutoff - continuing at get and save probability
##    ## separate training / validation data
##
##    probdata = np.load( _model_+'_prob.bin' )
##    prob_valid, prob_test = probdata['prob_valid'], probdata['prob_test']
##    print 'get prob_valid Y_valid:', prob_valid.shape, Y_valid.shape

prob_valid, Y_valid = prob_valid[:,1], Y_valid[:,1]
prob_test,  Y_test  = prob_test[:,1],  Y_test[:,1]

## get rates at various cutoff
result_table = np.zeros([len_search,4])
n = Y_valid.shape[0]*1.0
for cutoff in range(len_search):
    class_valid = prob_valid >= np.ones(prob_valid.size)*cutoff/float(len_search)
    n00 = sum(class_valid[Y_valid==0]==0)
    n01 = n - sum(Y_valid) - n00
    n10 = n - sum(class_valid) - n00
    result_table[cutoff,0] = cutoff / float(len_search)
    result_table[cutoff,1] = 1-(n01+n10)/n
    result_table[cutoff,2] = n00 / (n00+n10)
    result_table[cutoff,3] = n00 / (n00+n01)

## get cut_opt at conf_level
rate_table = np.zeros( (len(conf_level_in),2,3), dtype='float' )
for conf_level in conf_level_in:
    if print_rate: print '='*70
    i = conf_level_in.index(conf_level) #conf_level index
    
    target_index = result_table[:,2]>=conf_level #index of conf_rate >= conf_level
    ## if there is no optimal cutoff 
    if sum( target_index )==0:
        if print_rate: print 'No cutoff at', conf_level
        rate_table[i,:,:] = np.ones((2,3), dtype='float')*-1.
    else:
        result_table_2 = result_table[target_index,:]
        max_index = np.argwhere( result_table_2[:,3] == np.amax(result_table_2[:,3]) )
        cut_opt = np.mean( result_table_2[max_index,0] ) # optimal cutoff
        if print_rate: print 'optimal cuttoff with confidence level', conf_level, ':', cut_opt

        class_valid_opt = prob_valid >= np.ones(prob_valid.size)*cut_opt
        n00 = sum(class_valid_opt[Y_valid==0]==0)
        n01 = n - sum(Y_valid) - n00
        n10 = n - sum(class_valid_opt) - n00         
        rate_table[i,0,:] = [1-(n01+n10)/n, n00 / (n00+n10), n00 / (n00+n01)]
        
        ## get rates in test data
        class_test = prob_test >= cut_opt
        n_test = Y_test.shape[0]*1.0
        n00_test = sum(class_test[Y_test==0]==0)
        n01_test = n_test - sum(Y_test) - n00_test
        n10_test = n_test - sum(class_test) - n00_test
        rate_table[i,1,:] = [ 1-(n01_test+n10_test)/n_test, n00_test / (n00_test+n10_test), n00_test / (n00_test+n01_test) ]

    if print_rate:
        print ''
        print('-----rate table-----')
        print '%10s' % '', '%8s' % 'accuracy', '%10s' % 'confidence', '%9s' % 'detection'
        print '%10s' % 'validation', '%8.5f' % rate_table[i,0,0], '%10.5f' % rate_table[i,0,1], '%9.5f' % rate_table[i,0,2]
        print '%10s' % '   test   ', '%8.5f' % rate_table[i,1,0], '%10.5f' % rate_table[i,1,1], '%9.5f' % rate_table[i,1,2]
        print('')
        print('-----test data table-----')
        print '%8s' % 'pred'
        print 'true 0 ', '%5d'%sum(class_test[Y_test==0]==0), '%5d'%sum(class_test[Y_test==0]==1)
        print '     1 ', '%5d'%sum(class_test[Y_test==1]==0), '%5d'%sum(class_test[Y_test==1]==1)

