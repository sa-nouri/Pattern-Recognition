import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

NUM_OF_CLUSTERS = 2

def calc_separation_index(data, labels):
    num_of_clstrs= np.max(labels)+1

    down= -float("inf")
    for i in range(num_of_clstrs):
        data_i= data[labels==i,:]
        ni= data_i.shape[0]
        clstr_min_dist= float("inf")
        for j in range(ni):
            for k in range(ni):
                if ((j != k) and (np.linalg.norm(data_i[j]-data_i[k])<clstr_min_dist)):
                    clstr_min_dist= np.linalg.norm(data_i[j]-data_i[k])
        if (clstr_min_dist>down):
            down= clstr_min_dist
    
    up= float("inf")
    for i in range(num_of_clstrs):
        min_dist_ij= float("inf")
        for j in range(i):
            data_i= data[labels==i,:]
            data_j= data[labels==j,:]
            ni= data_i.shape[0]
            nj= data_j.shape[0]
            for k in range(ni):
                for l in range(nj):
                    if (np.linalg.norm(data_i[k]-data_j[l])<min_dist_ij):
                        min_dist_ij= np.linalg.norm(data_i[k]-data_j[l])
            if (min_dist_ij<up):
                up= min_dist_ij

    print ("HELLLLLLLLPPPP")
    print (up, down)
    return up/down


def calc_cov(np_arr):
    return np.matmul(np_arr.T,np_arr)

def calc_sw(data, labels):
    sum= 0
    for i in range(NUM_OF_CLUSTERS):
        mui= np.mean(data[labels==i, :],axis=0).reshape((1,-1))
        sum+= calc_cov(data[labels==i, :] - mui)
    return sum

def calc_sb(data, labels):
    sum= 0
    mu= np.mean(data, axis=0).reshape((1,-1))
    for i in range(NUM_OF_CLUSTERS):
        ni= len(np.where(labels==i)[0])
        mui= np.mean(data[labels==i, :],axis=0).reshape((1,-1))
        sum+= ni*calc_cov(mui - mu)
    return sum

def cluster_mean_distance(all_data, clstr_labels, label):
	clstr_data= all_data[clstr_labels==label,:]
	mu = np.mean(clstr_data, axis=0)
	return np.linalg.norm(clstr_data-mu)


# Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
all_labels = np.hstack((train_labels,test_labels))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size,
      'All Data Samples', all_data.shape)

#Affinity Clustering
all_data= train_data
clstr_model = AffinityPropagation(damping= 0.9, preference= -4500).fit(all_data)
clstr_labels = np.asarray(clstr_model.labels_, dtype=int)

for i in range(np.max(clstr_labels)+1):
    print("Distance of cluster ", str(i), ":", cluster_mean_distance(all_data, clstr_labels, i))

sw= calc_sw(all_data, clstr_labels)
sb= calc_sb(all_data, clstr_labels)

print("Norm of Sw-1Sb:", np.linalg.norm(np.linalg.inv(sw)*sb))

print("CVM: ", calc_separation_index(all_data, clstr_labels))

#PLOT
plt.hist(clstr_labels)
plt.show()
plt.hist(all_labels)
plt.show()

