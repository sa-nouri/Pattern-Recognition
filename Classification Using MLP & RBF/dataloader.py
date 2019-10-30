import numpy as np
from sklearn.feature_selection import VarianceThreshold

def select_features():
      # Loading Dataset
      data = np.loadtxt('data.csv', dtype=np.float32, delimiter=',')
      labels = np.loadtxt('labels.csv', dtype=np.int, delimiter=',')

      tr_samples_size = 245
      train_data = data[0:tr_samples_size,:]


      train_labels = labels[0:tr_samples_size]
      test_data = data[tr_samples_size:data.shape[0],:]
      test_labels = labels[tr_samples_size:data.shape[0]]
      class_names = ['0', '1', '2', '3', '4', '5', '6']

      tr_samples_size,_ = train_data.shape
      tr_samples_size, feature_size = train_data.shape
      te_samples_size, _ = test_data.shape

      return train_data, train_labels, test_data, test_labels, class_names, tr_samples_size, te_samples_size, len(class_names), feature_size
