import matplotlib.pyplot as plt
import numpy as np
from ex1_cost_function import *
#from ex2_derivative_function import *
from ex3_update_value import *
import time
import tensorflow as tf

def main():
	# Draw cost function
	x = np.arange(-10, 15, 0.1)
	y = [cost_function(i) for i in x]
	plt.plot(x, y)
	plt.ylabel('Cost Value')
	plt.xlabel('Input Variable')
	plt.title('Click on the figure to run Gradient Descent Algorithm')
	plt.waitforbuttonpress()

	# Run gradient descent
	l_rate = 0.01
	x = tf.Variable(-10, dtype = tf.float32)
	cost = 3*x**2 - 12*x + 4
	train = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	for i in range(100):
		sess.run(train)
		x_op = sess.run(x)
		plt.plot(x_op, cost_function(x_op), 'r+')
		plt.pause(0.1)

	print('Optimized variable: x_op = ', x_op)
	print('Optimized value: f(x_op)=', cost_function(x_op))	

if __name__ == '__main__':
	main()
