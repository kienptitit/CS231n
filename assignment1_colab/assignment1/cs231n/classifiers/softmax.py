from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]    
    num_classes = W.shape[1]
    y_pred = X.dot(W)
    loss = 0.0
    for i in range(num_train):
        y_i = y_pred[i]
        score_sm = np.exp(y_i) / np.sum(np.exp(y_i))
        loss += -np.log(score_sm[y[i]])
        
    loss /= num_train    
    loss += reg * np.sum(W * W)

    for i in range(num_train):
      for j in range(num_classes):
          dW[:,j] += X[i] * np.exp(X[i].dot(W[:,j])) / np.sum(np.exp(y_pred[i]))
          if j == y[i]:
             dW[:,j] += -X[i]

    dW /= num_train
    dW += reg * 2 * W
           
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    s = np.exp(X.dot(W))
    
    


    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    
    s_loss = s / s.sum(axis = -1,keepdims = True)
    loss = s_loss[np.arange(num_train),y]
    loss = -np.log(loss).mean()

    s_tmp = s / s.sum(axis = -1,keepdims = True)
    dW = -X.T.dot(one_hot) + X.T.dot(s_tmp)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
