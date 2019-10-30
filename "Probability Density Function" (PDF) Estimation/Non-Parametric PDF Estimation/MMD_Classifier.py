import itertools
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
from collections import Counter

num_of_nums= 10

import contextlib
import time
@contextlib.contextmanager
def timer():
    start= time.time()
    yield
    print("Elapssed time= "+ str(time.time()-start))

def min_mean_dist(x, td):
    means=[]
    for i in range(num_of_nums):
        idx= np.where(train_labels==i)[0]
        dists= np.sqrt(np.sum((td[idx]-x)**2, axis=1))
        means.append(np.mean(dists))
    return np.argmin(means)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
feat_size= feature_size

decision_labels= np.zeros(shape=(test_labels.shape[0],1))

with timer():
    for j in range(test_labels.shape[0]):
        decision_labels[j]= min_mean_dist(test_data[j], train_data)

print(np.sum(decision_labels.flat==test_labels.flat)/len(test_labels))

class_names = ['0', '1', '2','3','4','5','6','7','8','9']
cnf_matrix = confusion_matrix(test_labels, decision_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()
