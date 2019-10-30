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
import contextlib
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

@contextlib.contextmanager
def timer():
    start= time.time()
    yield
    print("Elapssed time= "+ str(time.time()-start))

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

knn = KNeighborsClassifier(n_neighbors=3)
with timer():
    knn.fit(train_data, train_labels)
with timer():
    decision_labels= knn.predict(test_data)
print(np.sum(decision_labels.flat==test_labels.flat)/len(test_labels))
class_names = ['0', '1', '2','3','4','5','6','7','8','9']
cnf_matrix = confusion_matrix(test_labels, decision_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()

parzen = RadiusNeighborsClassifier(radius=3.0)
with timer():
    parzen.fit(train_data, train_labels)
with timer():
    decision_labels= parzen.predict(test_data)
print(np.sum(decision_labels.flat==test_labels.flat)/len(test_labels))
class_names = ['0', '1', '2','3','4','5','6','7','8','9']
cnf_matrix = confusion_matrix(test_labels, decision_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()

gnb = GaussianNB()
with timer():
    gnb.fit(train_data, train_labels)
with timer():
    decision_labels= gnb.predict(test_data)
print(np.sum(decision_labels.flat==test_labels.flat)/len(test_labels))
class_names = ['0', '1', '2','3','4','5','6','7','8','9']
cnf_matrix = confusion_matrix(test_labels, decision_labels)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
plt.show()