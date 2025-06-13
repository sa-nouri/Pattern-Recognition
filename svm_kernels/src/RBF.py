### ===================================================== ###
### Pattern Recognition  -----  Assignment(5) ------ RBF ####
### Salar Nouri ----- 810194422 ###
### ===================================================== ###

import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')


## Data Initializing 
Data = pd.read_csv('Tiny Cancer Diagnosis dataset.csv', header = None )
Data_Labels = pd.read_csv('Cancer Diagnosis label.csv', header = None )

Train_Data = Data.iloc[0 : int(0.7 * len(Data)),:]
Train_labels = Data_Labels.iloc[ 0 : int(0.7 * len(Data))]
Test_Data = Data.iloc[int(0.7 * len(Data)) +1 :]
Test_labels = Data_Labels.iloc[int(0.7 * len(Data)) +1 :]

## Implementing the SVM algorithm with RBF kernel and finding the optimal parameters
parameters = {'C':[1,1.5,2,2.5,3], 'gamma':[0.17,0.18,0.19]}
svc=svm.SVC( kernel='rbf',decision_function_shape='ovo' )
clf = GridSearchCV(svc, parameters, return_train_score='False' )
clf.fit(Train_Data, Train_labels)
t3 = time.clock()
CCR = clf.score(Test_Data, Test_labels)
t4 = time.clock()

# clf1.cv_results_["params"][clf1.best_index_]
# CCR
# (clf1.cv_results_["mean_fit_time"][clf1.best_index_]+clf1.cv_results_["mean_score_time"][clf1.best_index_])
# (t4 - t3 )
print('Results For One Versus All Classifier')
print('Optimal Parameters : ')
print(clf.cv_results_["params"][clf.best_index_])
print('CCR : %f'%CCR)
print('Required time for Training the algorithm : %f'%(clf.cv_results_["mean_fit_time"][clf.best_index_]+clf.cv_results_["mean_score_time"][clf.best_index_]))
print('Required time for Testing the algorithm : %f'%(t4-t3))



#############################################################


#           RBF


C_=1.5
gamma_=0.18
Ctest=np.linspace(0.1,10)
gammatest=np.linspace(0.01,0.5)
CCR=np.zeros(np.size(Ctest))
for x in range (np.size(Ctest)):
    svc=svm.SVC(kernel='rbf',
                gamma=gamma_,
                C=Ctest[x])
    model=svc.fit(Train_Data,Train_labels)
    CCR[x]=model.score(Test_Data,Test_labels)
plt.plot(Ctest,CCR)
plt.title('CCR Corresponding to Different Values of C')
plt.xlabel('Value of Parameter C')
plt.ylabel('CCR')
plt.show()

CCR=np.zeros(np.size(gammatest))
for x in range (np.size(gammatest)):
    svc=svm.SVC(kernel='rbf',
                gamma=gammatest[x],
                C=C_)
    model=svc.fit(Train_Data,Train_labels)
    CCR[x]=model.score(Test_Data,Test_labels)
plt.plot(gammatest,CCR)
plt.title('CCR Corresponding to Different Values of Gamma')
plt.xlabel('Value of Parameter Gamma')
plt.ylabel('CCR')
plt.show()
