import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():
  A = tf.placeholder(dtype=tf.float32, shape=[1,2,3])
  B = A * 2
  C = A * 3

  a_numpy = np.random.rand(1,2,3)
  feed_dict = {A : a_numpy}

  sess = tf.Session()
  b_numpy,c_numpy = sess.run((B,C), feed_dict)

  print(a_numpy)
  print(b_numpy)
  print(c_numpy)
