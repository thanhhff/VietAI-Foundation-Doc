import numpy as np
import tensorflow as tf


def define_parameters():
    a = b = 0
    # TODO 1: Initialize parameters of model named: 'a' and 'b'
    a = tf.Variable(initial_value=tf.random_normal(shape=()), dtype=tf.float32, name='a')
    b = tf.Variable(initial_value=tf.random_normal(shape=()), dtype=tf.float32, name='b')
    return a, b


def define_cost_func(X, Y, a, b, n_sample):
    h = cost = 0
    # TODO 2: define hypothesis 'h' and cost function cost 'cost'
    
    h = a*X + b
    cost = tf.reduce_sum(tf.pow(h - Y, 2)) / (2*n_sample)
    return cost


def define_optimizer(l_rate, cost_func):
    optimizer = initializer = 0
    # TODO 3: define optimizer and initializer
    initializer = tf.global_variables_initializer()

    optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(cost_func)
    return optimizer, initializer
