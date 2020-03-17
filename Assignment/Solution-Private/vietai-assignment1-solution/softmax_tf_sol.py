"""
This file is for multiclass fashion-mnist classification using TensorFlow
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_mnist_data
from logistic_np_sol import add_one
from softmax_np_sol import *


if __name__ == "__main__":
    np.random.seed(2018)
    tf.set_random_seed(2018)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    # [TODO 2.8] Create weights (W) using TF variables
    w_shape = (train_x.shape[1],10)
    w = tf.Variable(np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape) , name = "w" , dtype=tf.float32)

    # [TODO 2.9] Create a feed-forward operator
    pred = tf.matmul(x, w)
    pred_max = tf.reduce_max(pred, 1, True)
    pred -= pred_max
    exp_pred = tf.exp(pred)
    pred = exp_pred / tf.reduce_sum(exp_pred, 1, True)

    # [TODO 2.10] Write the cost function
    cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(pred), 1))

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.01    

    # [TODO 2.8] Create an SGD optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            # [TODO 2.8] Update weights here
            train_loss = sess.run(cost, feed_dict={'x:0': train_x, 'y:0': train_y})
            val_loss = sess.run(cost, feed_dict={'x:0': val_x, 'y:0': val_y})
            sess.run(optimizer, feed_dict={'x:0': train_x, 'y:0': train_y})
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            # [TODO 2.11] Define your own stopping condition here
            if (len(all_val_loss)>2):
                if (all_val_loss[-1] > all_val_loss[-2]):
                    num_val_increase += 1

            if (num_val_increase > 10):
                break

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_train_loss, all_val_loss)
                w_  = sess.run(w)
                draw_weight(w_)
                plt.show()
                plt.pause(0.1)     
                print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss, val_loss))
        
        y_hat = sess.run(pred, feed_dict={'x:0': test_x})
        test(y_hat, test_y)
