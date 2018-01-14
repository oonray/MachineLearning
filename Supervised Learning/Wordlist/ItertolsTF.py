import tensorflow as tf
import numpy as np


mutations = 500

this = np.array([0,0,0])
last = np.array([0,0,0])

a = tf.placeholder("float")

d = a+1



with tf.Session() as sess:
    for i in range(mutations):
        last[0] = sess.run(d,feed_dict={a: this[0]})
        last[1] = sess.run(d, feed_dict={a: this[1]})
        last[2] = sess.run(d, feed_dict={a: this[2]})
        print(last)
        np.save("last.npy",last)
        this = last

