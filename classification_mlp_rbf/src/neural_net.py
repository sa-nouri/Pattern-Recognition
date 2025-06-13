import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def relu(x):
  return x*(x>0)

def relu_prime(x):
  return x>0

class MLPNet(object):

  """
  In this class we implement a MLP neural network.
  H: hidden layer size
  N: input size
  D: Number of features
  C: class
  Loss Function: Softmax
  Regularization: L2 norm
  Activation Function: ReLU

  """
  def __init__(self, D, H, output_size, std=1e-4):
    """
    In this part we initialize the model as below:
    weights are initialize with small random value and biases are initialized with zero value.
    these values are stored in the self.p_net as dictionary
    """
    self.p_net = {}
    self.p_net['W1'] = std * np.random.randn(D, H)
    self.p_net['b1'] = np.zeros(H)
    self.p_net['W2'] = std * np.random.randn(H, output_size)
    self.p_net['b2'] = np.zeros(output_size)

  def decode(self, N, C, y):
    C = self.p_net['W2'].shape[1]
    decoded_y= np.zeros(shape=(N,C))
    for i in range(N):
      decoded_y[i,y[i]]= 1
    return decoded_y 

  def loss(self, X, y=None, reg=0.0):

    """
      calculate the loss and its gradients for network:
      our inputs are:
        X: N*D matrix
        y: training labels

      Returns:
      if y is empty :
        -return score matrix with shape (N,C) .each element of this matrix shows score for class c on input X[i]
      otherwise:
        -return a tuple of loss and gradient.
    """

    N, D = X.shape
    H, C = self.p_net['W2'].shape

    X_anormal= X
    X_appended= X_anormal
    # X_appended= np.hstack([X_anormal, np.ones(shape=(N,1))])
    X = (X_appended-np.amin(X_appended, axis=0))/(np.amax(X_appended, axis=0)-np.amin(X_appended, axis=0))
    # print(X.shape)
    # X= X_appended/np.amax(X_appended, axis=0)
    Weight2_tmp, bias2 = self.p_net['W2'], self.p_net['b2']
    Weight1_tmp, bias1 = self.p_net['W1'], self.p_net['b1']

    #concatenation with biases
    # self.Weight1= np.vstack([Weight1_tmp, bias1])
    # self.Weight2= np.vstack([Weight2_tmp, bias2])
    self.Weight1= Weight1_tmp
    self.Weight2= Weight2_tmp
    
    # forward pass
    self.z1 = np.dot(X, self.Weight1) + bias1

    self.z2 = relu(self.z1)
    # self.z2 = np.hstack([self.z2, np.ones(shape=(N,1))])
    
    self.z3 = np.dot(self.z2, self.Weight2) + bias2
    
    y_hat = relu(self.z3)

    if y is None:
      return y_hat


    # fill loss function.
    y_decoded= self.decode(N, C, y)
    l2_distance= np.sum(np.asarray(y_decoded-y_hat)**2)
    reg_term= reg*(np.sum(np.asarray(self.Weight1)**2)+np.sum(np.asarray(self.Weight2)**2))
    loss= l2_distance + reg_term

    # calculate gradients
    gradient = {}
    
    self.o_error = y_decoded - y_hat
    self.o_delta = self.o_error*relu_prime(y_hat)

    self.z2_error = self.o_delta.dot(self.Weight2.T) 
    self.z2_delta = self.z2_error*relu_prime(self.z2) 

    g_w1= X.T.dot(self.z2_delta)
    g_w2= self.z2.T.dot(self.o_delta)

    # print("g_w1",g_w1.shape)
    # print("g_w2",g_w2.shape)
    # gradient['W1']= g_w1[0:-1,0:-1]
    # gradient['b1']= g_w1[-1,0:-1]
    # gradient['W2']= g_w2[0:-1]
    # gradient['b2']= g_w2[-1]
    gradient['W1']= g_w1 + 2*reg * self.Weight1
    gradient['b1']= np.sum(self.z2_delta, axis=0)
    gradient['W2']= g_w2 + 2*reg * self.Weight2
    gradient['b2']= np.sum(self.o_delta, axis=0)

    return loss, gradient

  def train(self, X, y, X_val, y_val,
            alpha=1e-3, alpha_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=100):

    """
    We want to train this network with stochastic gradient descent.
    Our inputs are:

    - X: array of shape (N,D) for training data.
    - y: training labels.
    - X_val: validation data.
    - y_val: validation labels.
    - alpha: learning rate
    - alpha_decay: This factor used to decay the learning rate after each epoch
    - reg: That shows regularization .
    - num_iters: Number of epoch
    - batch_size: Size of each batch

    """
    # print(y,"\n*******")
    num_train = X.shape[0]
    iteration = max(num_train / batch_size, 1)

    loss_train = []
    train_acc = []
    val_acc = []

    for it in range(num_iters):
      prm= np.random.permutation(X.shape[0])
      data_batch= X[prm[0:batch_size],:]
      label_batch= y[prm[0:batch_size]]
      # print(label_batch,"\n*****")
      # calculate loss and gradients
      loss, gradient = self.loss(data_batch, y=label_batch, reg=reg)
      loss_train.append(loss)

      self.p_net['W1'] += alpha * gradient['W1']
      self.p_net['W2'] += alpha * gradient['W2']
      self.p_net['b1'] += alpha * gradient['b1']
      self.p_net['b2'] += alpha * gradient['b2']

      if it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      if it % iteration == 0:
        # Check accuracy
        # print(label_batch, self.predict(data_batch), self.predict(data_batch) == label_batch)
        train_acc_new = (self.predict(data_batch) == label_batch).mean()
        # print(y_prediction, label_batch)
        val_acc_new = (self.predict(X_val) == np.array(y_val)).mean()
        train_acc.append(train_acc_new)
        val_acc.append(val_acc_new)

        alpha *= alpha_decay

    return {
      'loss_train': loss_train,
      'train_acc': train_acc,
      'val_acc': val_acc,
    }

  def predict(self, X):

    """
    After you train your network use its parameters to predict labels

    Returns:
    - y_prediction: array which shows predicted lables
    """
    y_hat= self.loss(X)
    y_prediction= np.argmax(y_hat, axis=1)

    return y_prediction
