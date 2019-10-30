import random as rnd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import KFold
import math
def main():
    from dataloader import select_features
    import numpy as np
    import os
    from scipy.misc import imread
    val_num = 45
    train_num = 245
    test_num = 40
    train_data, train_labels, test_data, test_labels,\
        class_names, n_train, n_test, n_class, n_features = select_features()
   # print(train_data)
    X_train = train_data
    #print(X_train)
    y_train = train_labels
    X_test = test_data
    y_test = test_labels
    # Subsample the data
    df_train = X_train
    df_test = X_test
    print('***' , df_test.shape)
    trueLabels = np.concatenate((y_train,y_test),axis=0)
    print ('Train data shape: ', X_train.shape)
    print ('Train labels shape: ', y_train.shape)
    print ('Test data shape: ', X_test.shape)
    print ('Test labels shape: ', y_test.shape)

    t=RBFNet()
    nlabels=7                #total number of labels
    nclusters=20              #number of clusters for k-means
    ksplits=5                #ksplits-fold cross validation
    #t.RBF()
    #############################################################################
    ###   Calculate optimal beta for this network and put it in to optimalBeta
    ###                              with crossvalidatin
    #############################################################################
    cross_val_data = X_train
    tr_num =0

    accuracies = np.zeros([1,10])
    for i in range(1 , 10):
        kf = KFold(n_splits=ksplits,shuffle=True)
        kfold_accu = np.zeros([1 , ksplits])
        beta_accu = np.zeros([1,10])
        kf_index = 0
        for train_index, test_index in kf.split(X_train):

            Xx_train, Xx_test = X_train[train_index], X_train[test_index]
            yy_train, yy_test = y_train[train_index], y_train[test_index]
            (predictedLabels_train_, centers_train_, centroidLabel_train_) = t.trainRBF(Xx_train, nclusters, i,
                                                                                    nlabels, yy_train)
            predictedTestLabels_ = t.RBF(Xx_test, i, centers_train_, centroidLabel_train_, nlabels)
            for j in range(len(predictedTestLabels_)):
                if(predictedTestLabels_[j] == yy_test[j]):
                     kfold_accu[0][kf_index] = kfold_accu[0][kf_index] + 1
            kfold_accu[0][kf_index] = (kfold_accu[0][kf_index]/len(predictedTestLabels_)) * 100
            kf_index = kf_index + 1
        beta_accu[0][i] = np.sum(kfold_accu[0]) / ksplits
    index_beta = np.argmax(beta_accu[0])
    optimalBeta = index_beta
    print(optimalBeta)



   # optimalBeta =None
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    #Train
    print('training...')
    (predictedLabels_train,centers_train,centroidLabel_train)=t.trainRBF(df_train,nclusters,optimalBeta,nlabels,trueLabels)

    #Test
    print('testing...')
    predictedTestLabels=t.RBF(df_test,optimalBeta,centers_train,centroidLabel_train,nlabels)

    #Test accuracy
    testLabels=y_test
    accuracy=0
    for y in range(len(predictedTestLabels)):
        if predictedTestLabels[y] == testLabels[y]:
            accuracy+=1
    accuracy=(accuracy/len(predictedTestLabels))*100
    print('Percent accuracy on test data:',accuracy)


class RBFNet():
    def __init__(self):
        pass

    def calcKmean(self,data,n):
        kmeanz = KMeans(n_clusters=n).fit(data)
        centers=np.array(kmeanz.cluster_centers_)
        closest,_=pairwise_distances_argmin_min(kmeanz.cluster_centers_,data)
        closest=np.array(closest)
        return (centers,closest)

    def RBF(self,data,beta,centers,centroidLabels,nlabels):
       # print(X_train)
        #############################################################################
        ### Train RBF to produce predicted labels
        ### you should return predictedlabels as shape of (N,)
        #############################################################################
       d , f = data.shape
       predictedLabels = np.zeros([d])
       cens = len(centers)
       temp = np.zeros([d,cens])
       for i in range(d):
           for j in range(cens):
                temp[i][j] = -1 * beta * ((np.linalg.norm(data[i]- centers[j]))**2)
                temp[i][j] = math.exp(temp[i][j])
       for i in range(d):
        maxs = np.zeros([cens,cens])
        for j in range(cens):
            maxs[int(centroidLabels[j])] = temp[i][j]
            predictedLabels[i] = np.argmax(maxs)




        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return predictedLabels

    def trainRBF(self,data,k,beta,nlabels,trueLabels):
        #k-means: Getting centroids and the row indices (in df_train) of the data points closest to the centroids
        t=RBFNet()
        (centers,indices)=t.calcKmean(data,k)
        #The label of each centroid according training data
        centroidLabel=np.zeros(len(centers))
        for x in range(len(centers)):
            centroidLabel[x]=trueLabels[indices[x]]
        predictedLabels=t.RBF(data,beta,centers,centroidLabel,nlabels)
        return (predictedLabels,centers,centroidLabel)

#
if __name__ == "__main__":
    main()
