"""dnn_np_sol.py
Solution of deep neural network implementation using numpy
Author: Kien Huynh 
Modified by : Phuong Hoang
"""

import numpy as np
import matplotlib.pyplot as plt
from util import *
from activation_np import *
from gradient_check import *
import pdb


class Config(object):
    def __init__(self, num_epoch=1000, batch_size=100, learning_rate=0.0005, momentum_rate=0.9, epochs_to_draw=10,
                 reg=0.00015, num_train=1000, visualize=True):
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.epochs_to_draw = epochs_to_draw
        self.reg = reg
        self.num_train = num_train
        self.visualize = visualize


class Layer(object):
    def __init__(self, w_shape, activation, reg=1e-5):
        """__init__

        :param w_shape: create w with shape w_shape using normal distribution
        :param activation: string, indicating which activation function to be used
        """
        mean = 0
        std = 1
        self.w = np.random.normal(0, np.sqrt(2. / np.sum(w_shape)), w_shape)
        self.activation = activation
        self.reg = reg

    def forward(self, x):
        """forward
        This function compute the output of this layer
        
        :param x: input
        """
        # [TODO 1.2]
        result = np.dot(x, self.w)

        # Compute different types of activation
        if (self.activation == 'sigmoid'):
            result = sigmoid(result)
        elif (self.activation == 'relu'):
            result = reLU(result)
        elif (self.activation == 'tanh'):
            result = tanh(result)
        elif (self.activation == 'softmax'):
            result = softmax_minus_max(result)

        self.output = result
        return result

    def backward(self, x, delta_prev):
        """backward
        This function compute the gradient of the loss function with respect to the parameter (w) of this layer

        :param x: input of the layer
        :param delta_prev: delta computed from the next layer (in feedforward direction) or previous layer (in backpropagation direction)
        """
        # [TODO 1.2]
        delta = delta_prev

        if (self.activation == 'sigmoid'):
            delta *= sigmoid_grad(self.output)
            w_grad = np.dot(x.T, delta)

        elif (self.activation == 'tanh'):
            delta *= tanh_grad(self.output)
            w_grad = np.dot(x.T, delta)

        elif (self.activation == 'relu'):
            delta *= reLU_grad(self.output)
            w_grad = np.dot(x.T, delta)

        # [TODO 1.4] Implement L2 regularization on weights here
        # lambd = np.exp(-3)
        w_grad += self.reg * self.w
        return w_grad, delta.copy()


class NeuralNet(object):
    def __init__(self, num_class=2, reg=1e-5):
        self.layers = []
        self.momentum = []
        self.reg = reg
        self.num_class = num_class

    def add_linear_layer(self, w_shape, activation):
        """add_linear_layer

        :param w_shape: create w with shape w_shape using normal distribution
        :param activation: string, indicating which activation function to be used
        """
        if (len(self.layers) != 0):
            if (w_shape[0] != self.layers[-1].w.shape[-1]):
                raise ValueError("Shape does not match between the added layer and previous hidden layer.")

        if (activation == 'sigmoid'):
            self.layers.append(Layer(w_shape, 'sigmoid', self.reg))
        elif (activation == 'relu'):
            self.layers.append(Layer(w_shape, 'relu', self.reg))
        elif (activation == 'tanh'):
            self.layers.append(Layer(w_shape, 'tanh', self.reg))
        elif (activation == 'softmax'):
            self.layers.append(Layer(w_shape, 'softmax', self.reg))
        self.momentum.append(np.zeros_like(self.layers[-1].w))

    def forward(self, x):
        """forward

        :param x: input
        """
        all_x = [x]
        for layer in self.layers:
            all_x.append(layer.forward(all_x[-1]))

        return all_x

    def compute_loss(self, y, s):
        """compute_loss
        Compute the average cross entropy loss using y (label) and s (predicted label scores)

        :param y:  the label, the actual class of the samples, in one-hot format
        :param s: the propabilities that the given samples belong to class k
        """

        # [TODO 1.3]
        # Estimating cross entropy loss from s and y 
        data_loss = - 1 / y.shape[0] * np.sum(y * np.log(s))

        # Estimating regularization loss from all layers
        reg_loss = 0  # self.reg / 2 * np.sum(w_last ** 2)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            reg_loss += np.sum(layer.w ** 2)
        reg_loss = reg_loss * self.reg / 2

        data_loss += reg_loss
        return data_loss

    def backward(self, y, all_x):
        """backward

        :param y:  the label, the actual class of the samples, in one-hot format
        :param all_x: input data and activation from every layer
        """

        # [TODO 1.5] Compute delta factor from the output

        delta = all_x[-1] - y
        delta /= y.shape[0]

        # [TODO 1.5] Compute gradient of the loss function with respect to w of softmax layer, use delta from the output

        # grad_last using regularization
        grad_last = np.dot(all_x[-2].T, delta) + self.reg * self.layers[-1].w

        grad_list = []
        grad_list.append(grad_last)

        for i in range(len(self.layers) - 1)[::-1]:
            prev_layer = self.layers[i + 1]
            layer = self.layers[i]
            x = all_x[i]

            # [TODO 1.5] Compute delta_prev factor for previous layer (in backpropagation direction)
            delta_prev = np.dot(delta, prev_layer.w.T)

            # Use delta_prev to compute delta factor for the next layer (in backpropagation direction)
            grad_w, delta = layer.backward(x, delta_prev)
            grad_list.append(grad_w.copy())

        grad_list = grad_list[::-1]
        return grad_list

    def update_weight(self, grad_list, learning_rate):
        """update_weight
        Update w using the computed gradient

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            grad = grad_list[i]
            layer.w = layer.w - learning_rate * grad

    def update_weight_momentum(self, grad_list, learning_rate, momentum_rate):
        """update_weight_momentum
        Update w using SGD with momentum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum_rate: float, momentum rate
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            self.momentum[i] = self.momentum[i] * momentum_rate + learning_rate * grad_list[i]
            layer.w = layer.w - self.momentum[i]


def test(s, test_y):
    """test
    Compute the confusion matrix based on labels and predicted values 

    :param s: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """
    # y_hat = s ## For dnn_tf.py
    if (s.ndim == 2):
        y_hat = np.argmax(s, axis=1)
    num_class = np.unique(test_y).size
    confusion_mat = np.zeros((num_class, num_class))

    for i in range(num_class):
        class_i_idx = test_y == i
        num_class_i = np.sum(class_i_idx)
        y_hat_i = y_hat[class_i_idx]
        for j in range(num_class):
            confusion_mat[i, j] = 1.0 * np.sum(y_hat_i == j) / num_class_i

    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::(num_class + 1)])


def unit_test_layer(your_layer):
    """unit test layer

    This function is used to test layer backward and forward for a random datapoint
    error < 1e-8 - you should be happy
    error > 1e-3  - probably wrong in your implementation
    """
    # generate a random data point
    x_test = np.random.randn(1, your_layer.w.shape[0])
    layer_sigmoid = Layer(your_layer.w.shape, your_layer.activation, reg=0.0)

    # randomize the partial derivative of the cost function w.r.t the next layer
    delta_prev = np.ones((1, your_layer.w.shape[1]))

    # evaluate the numerical gradient of the layer
    numerical_grad = eval_numerical_gradient(layer_sigmoid, x_test, delta_prev, False)

    # evaluate the gradient using back propagation algorithm
    layer_sigmoid.forward(x_test)
    # print('x_test:', x_test)
    # print(layer_sigmoid.forward(x_test))

    w_grad, delta = layer_sigmoid.backward(x_test, delta_prev)
    # print(numerical_grad.shape)
    # print out the relative error
    error = rel_error(w_grad, numerical_grad)
    print("Relative error between numerical grad and function grad is: %e" % error)


def minibatch_train(net, train_x, train_y, cfg):
    """minibatch_train
    Train your neural network using minibatch strategy

    :param net: NeuralNet object
    :param train_x: numpy tensor, train data
    :param train_y: numpy tensor, train label
    :param cfg: Config object
    """
    # [TODO 1.6] Implement mini-batch training
    all_loss = []
    # print(train_x.shape) (12000, 3)
    # print(train_y.shape) (12000,)
    # print(type(train_y))

    # mnist
    # print(train_x.shape) (50000, 785)
    # print(cfg.num_train[0])

    # print(cfg.num_train)

    num_train = 0

    # cfg.numtrain.ndim != 2
    if (isinstance(cfg.num_train, int)):
        num_train = cfg.num_train
    else:
        num_train = cfg.num_train[0]

    for e in range(cfg.num_epoch):
        shuffled_indices = np.random.permutation(num_train)
        train_x_shuffled = train_x[shuffled_indices]
        train_y_shuffled = train_y[shuffled_indices]
        # print(train_y_shuffled.shape) (12000, )

        sum_loss = 0
        for i in range(0, num_train, cfg.batch_size):
            train_set_x = train_x_shuffled[i:i + cfg.batch_size]
            train_set_y = train_y_shuffled[i:i + cfg.batch_size]
            # print(train_set_x.shape) (100, 3)
            # print(train_set_y.shape) (100, )

            train_set_y = create_one_hot(train_set_y, net.num_class)
            # train_set_y.shape (100, 3)  (3 group)

            all_x = net.forward(train_set_x)
            s = all_x[-1]
            loss = net.compute_loss(train_set_y, s)
            grads = net.backward(train_set_y, all_x)
            net.update_weight(grads, cfg.learning_rate)

            sum_loss += loss

        if (cfg.visualize and e % cfg.epochs_to_draw == cfg.epochs_to_draw - 1):
            s = net.forward(train_x[0::3])[-1]
            visualize_point(train_x[0::3], train_y[0::3], s)
            plot_loss(all_loss, 2)
            plt.show()
            # plt.pause(0.01)

        final_loss = sum_loss / cfg.batch_size
        all_loss.append(final_loss)
        print("Epoch %d: loss is %.5f" % (e + 1, final_loss))


def batch_train(net, train_x, train_y, cfg):
    """batch_train
    Train the neural network using batch SGD

    :param net: NeuralNet object
    :param train_x: numpy tensor, train data
    :param train_y: numpy tensor, train label
    :param cfg: Config object
    """

    train_set_x = train_x[:cfg.num_train].copy()
    train_set_y = train_y[:cfg.num_train].copy()
    train_set_y = create_one_hot(train_set_y, net.num_class)
    all_loss = []

    for e in range(cfg.num_epoch):
        all_x = net.forward(train_set_x)
        s = all_x[-1]
        loss = net.compute_loss(train_set_y, s)
        grads = net.backward(train_set_y, all_x)
        net.update_weight(grads, cfg.learning_rate)

        all_loss.append(loss)

        if (cfg.visualize and e % cfg.epochs_to_draw == cfg.epochs_to_draw - 1):
            s = net.forward(train_x[0::3])[-1]
            visualize_point(train_x[0::3], train_y[0::3], s)
            plot_loss(all_loss, 2)
            plt.show()
            # plt.pause(0.01)

        print("Epoch %d: loss is %.5f" % (e + 1, loss))


def bat_classification():
    # Load data from file
    # Make sure that bat.dat is in data/
    train_x, train_y, test_x, test_y = get_bat_data()

    # Show data
    # visualize_point(train_x, train_y, train_y)

    train_x, _, test_x = normalize(train_x, train_x, test_x)

    test_y = test_y.flatten()
    train_y = train_y.flatten()
    num_class = (np.unique(train_y)).shape[0]

    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)

    # Define hyper-parameters and train-related parameters
    cfg = Config(num_epoch=1000, learning_rate=0.001, num_train=train_x.shape[0])

    # Create NN classifier
    num_hidden_nodes = 100
    num_hidden_nodes_2 = 100
    num_hidden_nodes_3 = 100
    net = NeuralNet(num_class, cfg.reg)
    net.add_linear_layer((train_x.shape[1], num_hidden_nodes), 'relu')
    net.add_linear_layer((num_hidden_nodes, num_hidden_nodes_2), 'relu')
    net.add_linear_layer((num_hidden_nodes_2, num_hidden_nodes_3), 'relu')
    net.add_linear_layer((num_hidden_nodes_2, num_hidden_nodes_3), 'relu')
    net.add_linear_layer((num_hidden_nodes_3, num_class), 'softmax')

    # Sanity check - train in small number of samples to see the overfitting problem- the loss value should decrease rapidly
    # cfg.num_train = 500
    # batch_train(net, train_x, train_y, cfg)

    # Batch training - train all dataset
    # batch_train(net, train_x, train_y, cfg)

    # Minibatch training - training dataset using Minibatch approach
    minibatch_train(net, train_x, train_y, cfg)

    s = net.forward(test_x)[-1]
    test(s, test_y)


def mnist_classification():
    # Load data from file
    # Make sure that fashion-mnist/*.gz is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)
    train_x, val_x, test_x = normalize(train_x, train_x, test_x)

    num_class = (np.unique(train_y)).shape[0]

    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x)
    val_x = add_one(val_x)
    test_x = add_one(test_x)

    # Define hyper-parameters and train-related parameters
    cfg = Config(num_epoch=300, learning_rate=0.001, batch_size=200, num_train=train_x.shape, visualize=False)

    # Create NN classifier
    num_hidden_nodes = 100
    num_hidden_nodes_2 = 100
    num_hidden_nodes_3 = 100
    net = NeuralNet(num_class, cfg.reg)
    net.add_linear_layer((train_x.shape[1], num_hidden_nodes), 'relu')
    net.add_linear_layer((num_hidden_nodes, num_hidden_nodes_2), 'relu')
    net.add_linear_layer((num_hidden_nodes_2, num_hidden_nodes_3), 'relu')
    net.add_linear_layer((num_hidden_nodes_3, num_class), 'softmax')

    # Minibatch training - training dataset using Minibatch approach
    minibatch_train(net, train_x, train_y, cfg)
    # batch_train(net, train_x, train_y, cfg)

    s = net.forward(test_x)[-1]
    test(s, test_y)


if __name__ == '__main__':
    np.random.seed(2017)

    # numerical check for your layer feedforward and backpropagation
    your_layer = Layer((60, 100), 'sigmoid')
    unit_test_layer(your_layer)

    plt.ion()
    bat_classification()
    mnist_classification()

    # pdb.set_trace()
