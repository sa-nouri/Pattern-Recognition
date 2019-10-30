import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
##### Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#####Confusion Matrix Def
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##### Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)


#K=1####################################################################
knn = KNeighborsClassifier(n_neighbors=1)
t1=time.clock()
knn.fit(train_data,train_labels)
t2=time.clock()
CRR=knn.score(test_data, test_labels)
t3=time.clock()
print('CRR(k=1):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = knn.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1
dig=np.arange(10)
plot_confusion_matrix(confusion,dig,title='Confusion matrix K=1')  

#K=3####################################################################
knn = KNeighborsClassifier(n_neighbors=3)
t1=time.clock()
knn.fit(train_data,train_labels)
t2=time.clock()
CRR=knn.score(test_data, test_labels)
t3=time.clock()
print('CRR(k=3):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = knn.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1
dig=np.arange(10)
plt.figure()
plot_confusion_matrix(confusion,dig,title='Confusion matrix K=3') 

#K=5####################################################################
knn = KNeighborsClassifier(n_neighbors=5)
t1=time.clock()
knn.fit(train_data,train_labels)
t2=time.clock()
CRR=knn.score(test_data, test_labels)
t3=time.clock()
print('CRR(k=5):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = knn.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1

dig=np.arange(10)
plt.figure()
plot_confusion_matrix(confusion,dig,title='Confusion matrix K=5') 

#K=10####################################################################
knn = KNeighborsClassifier(n_neighbors=10)
t1=time.clock()
knn.fit(train_data,train_labels)
t2=time.clock()
CRR=knn.score(test_data, test_labels)
t3=time.clock()
print('CRR(k=10):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = knn.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1
dig=np.arange(10)
plt.figure()
plot_confusion_matrix(confusion,dig,title='Confusion matrix K=10') 

#------------------------------------------------------------------------------------------------#

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import RadiusNeighborsClassifier
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
##### Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#####Confusion Matrix Def
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##### Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)


#radius=1.0####################################################################
neigh = RadiusNeighborsClassifier(radius=2.9)
t1=time.clock()
neigh.fit(train_data,train_labels)
t2=time.clock()
CRR=neigh.score(test_data, test_labels)
t3=time.clock()
print('CRR(radius=2.9):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = neigh.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1
dig=np.arange(10)
plot_confusion_matrix(confusion,dig,title='Confusion matrix radius=2.9')  
#-------------------------------------------------------------------------------------------#

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
##### Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#####Confusion Matrix Def
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('')

#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##### Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)


#####################################################################
clf =  GaussianNB()
t1=time.clock()
clf.fit(train_data,train_labels)
t2=time.clock()
CRR=clf.score(test_data, test_labels)
t3=time.clock()
print('CRR(Naive):',CRR)
print('Training Time:',t2-t1)
print('Testing Time:',t3-t2)
predictions = clf.predict(test_data)
ok=0  
confusion=np.zeros((10,10))
confidence=np.zeros((10,10))
for i in range(test_data.shape[0]):
    confusion[test_labels[i],predictions[i]]+=1
dig=np.arange(10)
plot_confusion_matrix(confusion,dig,title='Confusion matrix Naive')  

#-------------------------------------------------------------------------------#
