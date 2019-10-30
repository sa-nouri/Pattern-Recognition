### ===================================================== ###
### Pattern Recognition  -----  Assignment(5) ------ Poly and Linear Kernel ####
### Salar Nouri ----- 810194422 ###
### ===================================================== ###

# Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time


plt.style.use('ggplot')

data = pd.read_csv('Iris.csv', header = None )
data = data.sample(frac=1).reset_index(drop=True)

## ===================================================== ## 
## ======= Main Function ====== ##
## ===================================================== ##

# Parameter Initialization
Train_Data = data.iloc[0 : int(( 0.7 * len(data))), 0:4]
Test_Data = data.iloc[int(( 0.7 * len(data))) : ,  0:4]
Train_Labels = data.iloc[0:int( 0.7 *len(data)), 4]
Test_Labels = data.iloc[int( 0.7 *len(data)) :, 4]


### Implementing the SVM algorithm and finding the optimal parameters for OVO decision function shape

parameters = {'C':[1,0.5,1.5,0.8,1.2],'degree':np.arange(1,4),'gamma':[0.1,0.05,0.15,0.08,0.12,0.2],'coef0':[0,0.05,0.1,0.15,0.2]}
svc = svm.SVC(kernel='poly',decision_function_shape='ovr')
clf=GridSearchCV(svc,parameters,return_train_score='False')
clf.fit(Train_Data, Train_Labels)
t1 = time.clock()
CCR = clf.score(Test_Data, Test_Labels)
t2 = time.clock()

print('Results For One Versus One Classifier')
print('Optimal Parameters : ')
print(clf.cv_results_["params"][clf.best_index_])
print('CCR : %f'%CCR)
print('Required time for Training the algorithm : %f'%(clf.cv_results_["mean_fit_time"][clf.best_index_]+clf.cv_results_["mean_score_time"][clf.best_index_]))
print('Required time for Testing the algorithm : %f'%(t2-t1))

# clf.cv_results_["params"][clf.best_index_]
# (clf.cv_results_["mean_fit_time"][clf.best_index_]+clf.cv_results_["mean_score_time"][clf.best_index_])
# (t2-t1) # Time speding 


## ============= Using function to scrutin the One Versus One Classifier with One Versus all Classifier

### +=================  Finish Part one ============== ###

# Plotting the goodness of fit curve for one versus one method and one versus all method

deg = np.arange(10) + 1
CCR = np.zeros(np.size(deg))

for i in range (np.size(deg)):
    clf = svm.SVC( C = 1.5 ,
                    coef0 = 0.2, 
                    gamma = 0.8, 
                    kernel = 'poly', 
                    degree = i + 1, 
                    decision_function_shape = 'ovo' ) # ovr
    model = clf.fit(Train_Data, Train_Labels)
    CCR[i] = model.score(Test_Data, Test_Labels)

# CCR , deg Printing 


## =============== Finish Part 2 ===================== ###
## =============== Part 3 should be explained ======== ###

parameters = {'C':[1,0.5,1.5,0.8,1.2]}
svc = svm.LinearSVC(multi_class = 'ovr') # ovr test consider too
clf = GridSearchCV(svc, parameters, return_train_score = 'False')
clf.fit(Train_Data, Train_Labels)
t1 = time.clock()
CCR = clf.score(Test_Data, Test_Labels)
t2 = time.clock()

# clf.cv_results_["params"][clf.best_index_]
# CCR
# (clf.cv_results_["mean_fit_time"][clf.best_index_]+clf.cv_results_["mean_score_time"][clf.best_index_])
# (t2-t1)

print('Results For One Versus Rest Classifier')
print('Optimal Parameters : ')
print(clf.cv_results_["params"][clf.best_index_])
print('CCR : %f'%CCR)
print('Required time for Training the algorithm : %f'%(clf.cv_results_["mean_fit_time"][clf.best_index_]+clf.cv_results_["mean_score_time"][clf.best_index_]))
print('Required time for Testing the algorithm : %f'%(t2-t1))
