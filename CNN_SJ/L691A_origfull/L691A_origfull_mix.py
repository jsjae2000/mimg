'''
GPU run command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sL691A_v08_mix.py
'''
import numpy as np
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, RMSprop
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, activity_l2
import theano
import yaml
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.externals import joblib

##print '='*120 +'\n' + '='*120
##print 'feat: my transform, pca & act: 31 layer, center 0'
##print '='*120 +'\n' + '='*120

np.random.seed(1337)  # for reproducibility
#print '\n'

#### input
## input - mix model
_model_ = 'L691A_origfull'
_featdata_ = '/home/nai/CNN_SJ/data_file/L290A_defect0405_feature.bin'
_actdata_ = '/home/nai/CNN_SJ/L290A_defect0405/' + _model_ + '_act31.bin'
#batch_size = 100
nb_epoch = 2000
_rfmodel_ = '/home/nai/CNN_SJ/L290A_defect0405/rf/'+_model_+'_rfmix31.pkl'
comp_p = 0.999


## input - search optimal cutoff
len_search = 5000 # length.out in (0,1)
conf_level_in = [0.98] # confidence level
batch_size_tun = 200 # batch_size

## input - svm



#### load feature data
## load feature data
feat_data = np.load( _featdata_ )
feat_train, feat_valid, feat_test = feat_data['feat_train'], feat_data['feat_valid'], feat_data['feat_test']
print 'load feat_train feat_valid feat_test:', feat_train.shape, feat_valid.shape, feat_test.shape

#### preprocessing
#n_train = round(feat_train.shape[0]*0.8); ind_train = np.arange(n_train, dtype='int')

## preprocessing 1: no standardization
## preprocessing 1_1: scailing from 0 to 1
zeroone_index  = np.array( [38,50,51] ) - 1
zeroone_scaler = preprocessing.MinMaxScaler()
zeroone_scaler.fit_transform(  feat_train[ :,zeroone_index ]  )
feat_train[ :,zeroone_index ] = zeroone_scaler.transform(  feat_train[ :,zeroone_index ]  )
feat_valid[ :,zeroone_index ] = zeroone_scaler.transform(  feat_valid[ :,zeroone_index ]  )
feat_test[ :,zeroone_index ]  = zeroone_scaler.transform(  feat_test[ :,zeroone_index ]  )

## preprocessing 1_2: with maximum absolute value
maxabs_index = np.array( [1,2,3,4,7,8,9,23,25,52,53,54,55,56,57,58,59] ) - 1
maxabs_scaler = preprocessing.MaxAbsScaler()
maxabs_scaler.fit_transform(  feat_train[ :,maxabs_index ]  )
feat_train[ :,maxabs_index ] = maxabs_scaler.transform(  feat_train[ :,maxabs_index ]  )
feat_valid[ :,maxabs_index ] = maxabs_scaler.transform(  feat_valid[ :,maxabs_index ]  )
feat_test[  :,maxabs_index ] = maxabs_scaler.transform(  feat_test[ :,maxabs_index ]  )

## preprocessing 2: standardization
## preprocessing 2_1: log transformation
purtur_index = np.array( [7,23,25] ) - 1
purtur_mag = np.array( [1.0, 0.1, 0.1] )
for j in purtur_index:
    feat_train[ feat_train[:,j]==0, j ] = purtur_mag[ np.where(purtur_index == j) ]
    feat_valid[ feat_valid[:,j]==0, j ] = purtur_mag[ np.where(purtur_index == j) ]
    feat_test [ feat_test[:,j]==0, j ]  = purtur_mag[ np.where(purtur_index == j) ]

log_index = np.array( [1,2,3,4,7,8,9,23,25,52,53] ) - 1
feat_train[ :,log_index ] = np.log(  feat_train[ :,log_index ]  )
feat_valid[ :,log_index ] = np.log(  feat_valid[ :,log_index ]  )
feat_test[ :,log_index ]  = np.log(  feat_test[ :,log_index ]  )

## preprocessing 2_2: standardization
std_index = np.delete( np.arange(feat_train.shape[1]), np.array([38,39,50,51,54,55,56,57,58,59,114])-1 )
std_scaler = preprocessing.StandardScaler().fit(  feat_train[ :,std_index ]  )
feat_train[:,std_index] = std_scaler.transform( feat_train[:,std_index] )
feat_valid[:,std_index] = std_scaler.transform( feat_valid[:,std_index] )
feat_test[:,std_index]  = std_scaler.transform( feat_test[:,std_index]  )
print 'preprocessing feat_train feat_valid feat_test: complete'
print 'load feat_train feat_valid feat_test:', feat_train.shape, feat_valid.shape, feat_test.shape

##std_scaler = preprocessing.StandardScaler().fit(  feat_train[ ind_train,: ]  )
##feat_train = std_scaler.transform( feat_train )
##feat_test  = std_scaler.transform( feat_test  )

feat_pca = PCA(whiten=False)
feat_pca.fit( feat_train )
feat_train = feat_pca.transform( feat_train )
feat_valid = feat_pca.transform( feat_valid )
feat_test  = feat_pca.transform( feat_test )



#### load activation and feature data
## load activation & Y data
act_data = np.load( _actdata_ )
act_train, act_valid, act_test = act_data['act_train'], act_data['act_valid'], act_data['act_test']
Y_train, Y_valid, Y_test = act_data['Y_train'], act_data['Y_valid'], act_data['Y_test']
print 'load act_train act_valid act_test:', act_train.shape, act_valid.shape, act_test.shape
print 'load Y_train Y_valid Y_test:', Y_train.shape, Y_valid.shape, Y_test.shape

##act_train = act_train.reshape(act_train.shape[0], act_train.shape[1]*act_train.shape[2]*act_train.shape[3] )
##act_test  = act_test.reshape(act_test.shape[0], act_test.shape[1]*act_test.shape[2]*act_test.shape[3] )
##print '16: reshape act_train act_test:', act_train.shape, act_test.shape

##act_mean = np.mean( act_train[ind_train,:], axis=0)
##act_train -= act_mean
##act_test -= act_mean
pca = PCA(n_components=comp_p)
pca.fit( act_train )
act_train = pca.transform( act_train )
act_valid = pca.transform( act_valid )
act_test  = pca.transform( act_test )
print 'PCA for act_train act_valid act_test:', act_train.shape, act_valid.shape, act_test.shape



#### get comp data
comp_train, comp_valid, comp_test = np.append(act_train, feat_train, axis=1), np.append(act_valid, feat_valid, axis=1), np.append(act_test, feat_test, axis=1)
print 'get comp_train comp_valid comp_test', comp_train.shape, comp_valid.shape, comp_test.shape, '\n'

##comp_pca = PCA(whiten=False)
##comp_pca.fit( comp_train[ind_train,:] )
##comp_train = comp_pca.transform( comp_train )
##comp_test  = comp_pca.transform( comp_test )


#### cutoff tunning function
def _tunning_(prob_valid, prob_test, Y_valid, Y_test, len_search, conf_level_in, print_rate):
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

    ## get cutoff and rates at validation / test data
    rate_table = np.ones( (len(conf_level_in),2,3), dtype='float' )*-1.
    confusion_table = np.ones( (len(conf_level_in),4), dtype='int' )*-1
    cutoff_array = np.ones( len(conf_level_in) )*-1.
    for conf_level in conf_level_in:
        if print_rate: print '='*70
        i = conf_level_in.index(conf_level) #conf_level index
        
        target_index = result_table[:,2]>=conf_level #index of conf_rate >= conf_level
        ## if there is no optimal cutoff 
        if sum( target_index )==0:
            if print_rate: print 'No cutoff at', conf_level
        else:
            result_table_2 = result_table[target_index,:]
            max_index = np.argwhere( result_table_2[:,3] == np.amax(result_table_2[:,3]) )
            cut_opt = np.amax( result_table_2[max_index,0] ) # optimal cutoff
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

            ## get results
            rate_table[i,1,:] = [ 1-(n01_test+n10_test)/n_test, n00_test / (n00_test+n10_test), n00_test / (n00_test+n01_test) ]
            confusion_table[i,:] = np.array( [n00_test, n10_test, n01_test, sum(class_test[Y_test==1]==1)], dtype='int') 
            cutoff_array[i] = cut_opt
            
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

    return [rate_table, confusion_table, cutoff_array]



#### model save load cutoff print function
def _modelling_(model):
    ## save the model
    model_yaml = model.to_yaml(); del(model)
    open( _model_+'_mix'+ str(iter_model) +'.yaml', 'w' ).write(model_yaml)
    print 'save', _model_+'_mix'+ str(iter_model) +'.yaml'
    
    ## save History
    with open( _model_+'_mixhistory'+ str(iter_model) +'.h5','w' ) as fp:
        yaml.dump(hist.history, fp)
    print 'save',  _model_+'_mixhistory'+ str(iter_model) +'.h5 \n'



    #### load model - use at activation & probability & cutoff
    model = model_from_yaml(open( _model_+'_mix'+ str(iter_model) +'.yaml' ).read())
    model.load_weights( _model_+'_mix'+ str(iter_model) +'.h5' )
    print 'load', _model_, '\n'



    #### search optimal cutoff
    ## save probability
    #print('get probability')
    #prob_train = model.predict_proba(data_train, batch_size=batch_size_tun)
    #print 'get prob_train:', prob_train.shape
    prob_valid = model.predict_proba(data_valid, batch_size=batch_size_tun)
    #print 'get prob_train:', prob_train.shape
    prob_test  = model.predict_proba(data_test, batch_size=batch_size_tun)
    #print 'get prob_test:', prob_test.shape

    ## separate training / validation data
    score = model.evaluate(data_valid, Y_valid, show_accuracy=True, verbose=0) # get valid
    print '\n','='*70
    print 'validation loss validation accuracy:', round(score[0],5), round(score[1],5)
    
    #prob_valid = np.delete(prob_train, ind_train, 0)
    #print 'get prob_valid Y_valid:', prob_valid.shape, Y_valid.shape
    prob_valid2, Y_valid2 = prob_valid[:,1], Y_valid[:,1]
    prob_test2,  Y_test2  = prob_test[:,1],  Y_test[:,1]

    ## search optimal cutoff
    tun_result = _tunning_(prob_valid2, prob_test2, Y_valid2, Y_test2, len_search, conf_level_in, print_rate=True)
    result = np.append( tun_result[0].reshape(len(conf_level_in),6), tun_result[1], axis=1)
    result = np.append( result, tun_result[2].reshape(len(conf_level_in),1), axis=1)
    return [score, result]

#########################################################################################
#########################################################################################
#########################################################################################
###### SVM
##svm = SVC(probability=True)
##Y_train_svm = Y_train[:,1]
##svm.fit(comp_train, Y_train_svm)
##joblib.dump(svm, _svmmodel_)
##print 'save model at', _svmmodel_
##
##svm = joblib.load(_svmmodel_)
##print 'load modal:', _svmmodel_ 
##prob_valid = svm.predict_proba( comp_valid )
##prob_test  = svm.predict_proba( comp_test )
##                    
##print 'get prob_valid Y_valid:', prob_valid.shape, Y_valid.shape
##prob_valid2, Y_valid2 = prob_valid[:,1], Y_valid[:,1]
##prob_test2,  Y_test2  = prob_test[:,1],  Y_test[:,1]
##
#### search optimal cutoff
##tun_result = _tunning_(prob_valid2, prob_test2, Y_valid2, Y_test2, len_search, conf_level_in, print_rate=True)


#### random forest
for n_tree in np.random.random_integers(10,2000,5):
    print 'n_tree:', n_tree 
    rf = RandomForestClassifier(n_estimators=n_tree)
    Y_train_rf = Y_train[:,1]
    rf.fit(comp_train, Y_train_rf)
    joblib.dump(rf, _rfmodel_)
    print 'save model at', _rfmodel_

    rf = joblib.load(_rfmodel_)
    print 'load modal:', _rfmodel_ 
    prob_valid = rf.predict_proba( comp_valid )
    prob_test  = rf.predict_proba( comp_test )
                        
    print 'get prob_valid Y_valid:', prob_valid.shape, Y_valid.shape
    prob_valid2, Y_valid2 = prob_valid[:,1], Y_valid[:,1]
    prob_test2,  Y_test2  = prob_test[:,1],  Y_test[:,1]

    ## search optimal cutoff
    tun_result = _tunning_(prob_valid2, prob_test2, Y_valid2, Y_test2, len_search, conf_level_in, print_rate=True)
