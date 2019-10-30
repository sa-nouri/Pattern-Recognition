import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

NUM_OF_CLASSES= 7

def main():
    data= np.loadtxt('/Users/alish/Desktop/HW4/Cancer/data.csv', delimiter=',', dtype= int)
    labels= np.loadtxt('/Users/alish/Desktop/HW4/Cancer/labels.csv', delimiter=',', dtype= int)

    data_nrm= (data - np.mean(data,axis= 0))#/np.sqrt(np.var(data,axis= 0))
    # data_nrm = data

    tmp_eig_vectors, eig_values, _= np.linalg.svd(np.matmul(data_nrm,data_nrm.T))

    tmp_eig_vectors= tmp_eig_vectors[:, :-1]
    eig_values= eig_values[:-1]

    eig_vectors= np.matmul(np.transpose(data_nrm), tmp_eig_vectors)
    data_pca= np.matmul(np.transpose(eig_vectors),np.transpose(data_nrm))
    data_pca_t= np.transpose(data_pca)

    sw= calc_sw(data_pca_t, labels)
    sb= calc_sb(data_pca_t, labels)
    sepmat= np.matmul(inv(sw),sb)

    tmp_eig_vectors_sepmat, eig_values_sepmat, _= np.linalg.svd(sepmat)
    
    eig_vectors_sepmat= np.matmul(np.transpose(sepmat), tmp_eig_vectors_sepmat)
    sepmat_pca= np.matmul(np.transpose(eig_vectors_sepmat),np.transpose(sepmat))
    sepmat_pca_t= np.transpose(sepmat_pca)

    # plt.loglog(eig_values_sepmat)
    # plt.show()
    # print(eig_values_sepmat)

    # cum_sum= np.zeros(shape=(len(eig_values),1))
    # cum_sum[0]= np.sum(eig_values)
    # for i in reversed(range(len(eig_values))):
    #     cum_sum[i]= np.sum(eig_values[:i])
    cum_sum= list(reversed(np.cumsum(eig_values_sepmat)))+[0]
    plt.clf()
    plt.plot(cum_sum)
    plt.show()


def calc_cov(np_arr):
    return np.matmul(np_arr.T,np_arr)

def calc_sw(data_pca_t, labels):
    sum= 0
    for i in range(NUM_OF_CLASSES):
        mui= np.mean(data_pca_t[labels==i, :],axis=0).reshape((1,-1))
        sum+= calc_cov(data_pca_t[labels==i, :] - mui)
    return sum

def calc_sb(data_pca_t, labels):
    sum= 0
    mu= np.mean(data_pca_t, axis=0).reshape((1,-1))
    for i in range(NUM_OF_CLASSES):
        ni= len(np.where(labels==i)[0])
        mui= np.mean(data_pca_t[labels==i, :],axis=0).reshape((1,-1))
        sum+= ni*calc_cov(mui - mu)
    return sum


if __name__ == '__main__':
    main()