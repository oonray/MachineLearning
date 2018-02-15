from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def nn_model(data):
    data = tf.reshape(data, shape=[-1, 28, 28, 1])

    CW = tf.Variable(tf.random_normal([5,5,1,32]))
    CB = tf.Variable(tf.random_normal([32]))

    one = tf.nn.relu(tf.nn.conv2d(data, tf.add(CW,CB), strides=[1, 1, 1, 1], padding="SAME"))
    one = tf.nn.max_pool(one,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    C2W = tf.Variable(tf.random_normal([5,5,32,64]))
    C2B = tf.Variable(tf.random_normal([64]))

    two = tf.nn.relu(tf.nn.conv2d(one, tf.add(C2B,C2W), strides=[1, 1, 1, 1], padding="SAME"))
    two = tf.nn.max_pool(two,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    FULLW = tf.Variable(tf.random_normal([7*7*64,1024]))
    FULLB = tf.Variable(tf.random_normal([1024]))

    full = tf.nn.relu(tf.add(tf.matmul(tf.reshape(two,[-1, 7*7*64]),FULLW),FULLB))

    OUTW = tf.Variable(tf.random_normal([1024,10]))
    OUTB = tf.Variable(tf.random_normal([1024]))

    return tf.add(tf.matmul(full,OUTW),OUTB)


