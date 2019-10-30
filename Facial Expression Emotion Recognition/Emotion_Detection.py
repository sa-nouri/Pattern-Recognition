import scipy.misc
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as sn
import scipy.linalg as sl
def main():
    train_data , train_label = loadImages('train')
    test_data , test_label = loadImages('test')

    print("number of train data images is" , train_data.shape[0] , "and number of features for each image is", train_data.shape[1])

    train_data_normalized= train_data - np.mean(train_data, axis=0)
    test_data_normalized= test_data - np.mean(train_data, axis=0)

#     eig_values, tmp_eig_vectors = np.linalg.eig(np.matmul(train_data_normalized,train_data_normalized.T))
#     eig_values, tmp_eig_vectors = np.linalg.eig(np.cov(train_data))
    tmp_eig_vectors, eig_values, tmp_eig_vectors_trs= np.linalg.svd(np.matmul(train_data_normalized,train_data_normalized.T))
    print(np.sort(eig_values))
    tmp_eig_vectors= tmp_eig_vectors[:, :-1]
    eig_values= eig_values[:-1]

    eig_vectors= np.matmul(np.transpose(train_data), tmp_eig_vectors )
    
    train_data_pca= np.matmul(np.transpose(eig_vectors),np.transpose(train_data_normalized))
    d_i_sqr=np.linalg.inv(sl.sqrtm(np.diag(eig_values)))
    train_data_pca_white= np.matmul(d_i_sqr,train_data_pca)
    # train_data_pca_white= train_data_pca

    test_data_pca= np.matmul(np.transpose(eig_vectors),np.transpose(test_data_normalized))
    test_data_pca_white= np.matmul(d_i_sqr,test_data_pca)
    # test_data_pca_white= test_data_pca


    print(np.sort(eig_values))
    knn= sn.KNeighborsClassifier(n_neighbors=1)
    knn.fit(np.transpose(train_data_pca_white), train_label)
    predicts= knn.predict(np.transpose(test_data_pca_white))
    print(np.mean(predicts==test_label))

    knn.fit(train_data, train_label)
    predicts= knn.predict(test_data)
    print(np.mean(predicts==test_label))
    

    plt.plot(eig_values)
    plt.show()
    plt.loglog(eig_values)
    plt.show()
    plt.imshow(eig_vectors[:,0].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,1].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,2].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,3].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,152].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,151].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,150].reshape(256,256) , cmap='gray')
    plt.show()
    plt.imshow(eig_vectors[:,149].reshape(256,256) , cmap='gray')
    plt.show()

def loadImages(dirName):    
    # This function loads images from any directory
    # :param str dirName: is address of the directory (string)
    data = []
    label = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            face = scipy.misc.imread(os.path.join(root, file)) # Load image from a path
            face = face.reshape(256 * 256, ).tolist()          # Flatten image . Note: size of any image is 256,256
            data.append(face)
            label.append(file.split('.')[1])                   # Label of an image is in its fileName
    return np.asarray(data) , label

if __name__ == '__main__':
    main()