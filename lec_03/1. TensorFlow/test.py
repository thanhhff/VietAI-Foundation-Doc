import tensorflow as tf

tf.compat.v1.enable_eager_execution()
node_two = tf.constant(2)
node_three = tf.constant(3)
sum_node = node_two + node_three
print(sum_node)