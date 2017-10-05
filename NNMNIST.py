from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('./tmp', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

Weights = tf.Variable(tf.zeros([784,10]))
Biases = tf.Variable(tf.zeros([10]))

with tf.Session() as sess:
    pass

