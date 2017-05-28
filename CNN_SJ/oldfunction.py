
#### cutoff tunning function
def _tunning_(model, data_train, data_test, Y_train, Y_test, batch_size_tun):
        #### Save probability
        print('get probability')
        prob_train = model.predict_proba(data_train, batch_size=batch_size_tun)
        print 'get prob_train:', prob_train.shape
        prob_test  = model.predict_proba(data_test, batch_size=batch_size_tun)
        print 'get prob_test:', prob_test.shape

        #### search optimal cutoff
        ## separate training / validation data
        n_train = round(act_train.shape[0]*0.8)
        ind_train = np.arange(n_train)
        prob_valid = np.delete(prob_train, ind_train, 0)
        Y_valid = np.delete(Y_train, ind_train, 0)
        print 'get prob_valid Y_valid:', prob_valid.shape, Y_valid.shape

        prob_valid, Y_valid = prob_valid[:,1], Y_valid[:,1]
        prob_test,  Y_test  = prob_test[:,1],  Y_test[:,1]

        ## get rates at various cutoff
        result_table = np.zeros([len_search,4])
        n = Y_valid.shape[0]*1.0
        for cutoff in range(1,len_search):
                classes = prob_valid >= np.ones(prob_valid.size)*cutoff/float(len_search)
                n11 = sum(classes[Y_valid==1]==1)
                n10 = sum(Y_valid) - n11
                n01 = sum(classes) - n11
                n00 = n - n01 - n10 - n11
                result_table[cutoff,0] = cutoff / float(len_search)
                result_table[cutoff,1] = (n00 + n11) / n
                result_table[cutoff,2] = n00 / (n00 + n10) 
                result_table[cutoff,3] = n00 / (n00 + n01)

        ## get cut_opt at conf_level
        for conf_level in conf_level_in:
                if sum(result_table[:,2]>=conf_level)==0:
                        print 'No cutoff at', conf_level
                        continue
                ind_opt = np.argmax(result_table[result_table[:,2]>=conf_level,3])
                cut_opt = result_table[result_table[:,2]>=conf_level, 0][ind_opt]

                ## optimal cut off
                print '-'*40
                print 'optimal cuttoff with confidence level', conf_level, ':', cut_opt
                print 'In validation data'
                print('---rates---')
                print 'accuracy rate:,', result_table[result_table[:,2]>=conf_level, 1][ind_opt]
                print 'confidence rate:', result_table[result_table[:,2]>=conf_level, 2][ind_opt]
                print 'detection rate:', result_table[result_table[:,2]>=conf_level, 3][ind_opt]

                ## In test data
                Y_true = Y_test
                classes_test = prob_test >= cut_opt
                print('')
                print('In test data')
                print('---rates---')
                print 'accuracy rate:,', (sum(classes_test[Y_true==0]==0) + sum(classes_test[Y_true==1]==1))/(classes_test.size*1.0) 
                print 'confidence rate:', sum(classes_test[Y_true==0]==0) / (sum(classes_test==0)*1.0)
                print 'detection rate:', sum(classes_test[Y_true==0]==0) / (sum(Y_true==0)*1.0)

                print('')
                print('---table---')
                print('        pred')
                print 'true 0 ', sum(classes_test[Y_true==0]==0), sum(classes_test[Y_true==0]==1)
                print '     1 ', sum(classes_test[Y_true==1]==0), sum(classes_test[Y_true==1]==1)









## not efficient for multiple conf_level
#### cutoff tunning function
def _tunning_(model, prob_valid, prob_test, Y_valid, Y_test, len_search, conf_level, print_rate):
    ## search optimal cutoff
    #print '='*70
    cutoff = 1.0
    n = Y_valid.shape[0]*1.0
    while( cutoff>0 ):
        class_valid = prob_valid >= (np.ones(prob_valid.size)*cutoff)
        n00 = sum(class_valid[Y_valid==0]==0)
        n01 = n - sum(Y_valid) - n00
        n10 = n - sum(class_valid) - n00
        if n00 / (n00 + n10) >= conf_level: break
        cutoff -= 1.0/len_search
        
    rate_table = np.zeros( (2,3), dtype='float' )
    if cutoff==0 or n00+n10==0:
        if print_rate: print 'no optimal cutoff at level', conf_level
        rate_table = rate_table -1
        return(rate_table)
    else:
        if print_rate: print 'optimal cuttoff with confidence level', conf_level, ':', cutoff
        rate_table[0,:] = [ 1-(n01+n10)/n, n00 / (n00+n10), n00 / (n00+n01) ]

        ## get rates in test data
        class_test = prob_test >= cutoff
        n = Y_test.shape[0]*1.0
        n00 = sum(class_test[Y_test==0]==0)
        n01 = n - sum(Y_test) - n00
        n10 = n - sum(class_test) - n00
        rate_table[1,:] = [ 1-(n01+n10)/n, n00 / (n00+n10), n00 / (n00+n01) ]

        if print_rate:
            print ''
            print('-----rate table-----')
            print '%10s' % '', '%8s' % 'accuracy', '%10s' % 'confidence', '%9s' % 'detection'
            print '%10s' % 'validation', '%8.5f' % rate_table[0,0], '%10.5f' % rate_table[0,1], '%9.5f' % rate_table[0,2]
            print '%10s' % '   test   ', '%8.5f' % rate_table[1,0], '%10.5f' % rate_table[1,1], '%9.5f' % rate_table[1,2]
            print('')
            print('-----test data table-----')
            print '%8s' % 'pred'
            print 'true 0 ', '%5d'%sum(class_test[Y_test==0]==0), '%5d'%sum(class_test[Y_test==0]==1)
            print '     1 ', '%5d'%sum(class_test[Y_test==1]==0), '%5d'%sum(class_test[Y_test==1]==1)
        return rate_table
