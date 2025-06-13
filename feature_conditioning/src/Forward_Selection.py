import numpy as np
import sklearn.naive_bayes as sn
import matplotlib.pyplot as plt

NUM_OF_TRAIN= 200

def calc_CCR(train_data, train_label, test_data, test_label):
    gnb= sn.GaussianNB()
    gnb.fit(train_data, train_label)
    return np.mean(test_label==gnb.predict(test_data))

def main():
    data= np.loadtxt('/Users/alish/Desktop/HW4/Cancer/data.csv', delimiter=',', dtype= int)
    labels= np.loadtxt('/Users/alish/Desktop/HW4/Cancer/labels.csv', delimiter=',', dtype= int)
    train_data= data[:NUM_OF_TRAIN, :]
    test_data= data[NUM_OF_TRAIN:, :]
    train_label= labels[:NUM_OF_TRAIN]
    test_label= labels[NUM_OF_TRAIN:]

    train_data_nrm= (train_data - np.mean(train_data,axis= 0))/np.sqrt(np.var(train_data,axis= 0))
    test_data_nrm= (test_data - np.mean(train_data,axis= 0))/np.sqrt(np.var(train_data,axis= 0))
    
    ccrs= []
    num_feats= train_data.shape[1]
    selected= []
    is_selected= np.zeros(shape=(num_feats,1))
    for i in range(0, 200):
        candid_selected= selected
        best_ccr= 0
        for j in range(0,num_feats):
            if (is_selected[j]==0):
                candid_selected= selected+[j]
                curr_ccr= calc_CCR(train_data_nrm[:,candid_selected], train_label, test_data_nrm[:,candid_selected], test_label)
                if (curr_ccr>best_ccr):
                    best_id= j
                    best_ccr= curr_ccr
        is_selected[best_id]= 1
        selected= selected+[best_id]
        ccrs.append(best_ccr)
        print(best_id)
        print(best_ccr)

    print(selected)
    plt.plot(ccrs)
    plt.show()
    
if __name__ == '__main__':
    main()