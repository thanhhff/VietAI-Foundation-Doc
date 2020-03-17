import matplotlib.pyplot as plt
import numpy as np
from ex1_cost_function import *
from ex2_derivative_function import *
from ex3_update_value import *
import time

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
	eps = 0.001
	l_rate = 0.01
	x_op = -10

	while True:
		x_new = update_optimized_var(x_op, l_rate)

		# Check break point
		if abs(x_new-x_op) < eps:
			break
		# Update value of x_op with x_new
		x_op = x_new
		# Draw updated point
		plt.plot(x_op, cost_function(x_op), 'r+')
		plt.pause(0.1)

	print('Optimized variable: x_op = ', x_op)
	print('Optimized value: f(x_op)=', cost_function(x_op))	

if __name__ == '__main__':
	main()
