"""activation_np_sol.py
This file provides activation functions for the NN 
Author: Phuong Hoang
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    Sigmoid function. Output = 1 / (1 + exp(-1)).
    :param x: input
    """

    x = 1/(1+np.exp(-x))
    return x


def sigmoid_grad(a):
    """sigmoid_grad
    Compute gradient of sigmoid.
    :param a: output of the sigmoid function
    """
    
    return (a)*(1-a)


def reLU(x):
    """reLU
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    
    return np.maximum(0,x)


def reLU_grad(a):
    """reLU_grad
    Compute gradient of ReLU.
    :param x: output of ReLU
    """

    grad = np.copy(a)
    grad[grad <= 0] = 0
    grad[grad > 0] = 1
    return grad


def tanh(x):
    """tanh
    Tanh function.
    :param x: input
    """
   
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def tanh_grad(a):
    """tanh_grad
    Compute gradient for tanh.
    :param a: output of tanh
    """

    return 1 - a**2


def softmax(x):
    """softmax
    Softmax function.
    :param x: input
    """

    exp_scores = np.exp(x)
    probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)
    return probs


def softmax_minus_max(x):
    """softmax_minus_max
    Stable softmax function.
    :param x: input
    """

    exp_scores = np.exp(x - np.max(x, axis = 1, keepdims = True))
    probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)
    return probs 
