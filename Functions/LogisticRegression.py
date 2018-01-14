import tensorflow as tf
import numpy as np

x = tf.placeholder("float")
y = tf.placeholder("float")

x_data = np.random.rand(100).astype(np.float32)

y_data = (x_data*3)+2
y_data = np.vectorize(lambda y: y+np.random.normal(0.0,0.1))(y_data)

W = tf.Variable(1.0)
B = tf.Variable(0.2)

y = (x_data*W)+B

loss = tf.reduce_mean(tf.square(y-y_data))

optimiser = tf.train.GradientDescentOptimizer(0.5)

train = optimiser.minimize(loss)

init = tf.initialize_all_variables()

train_data=[]
with tf.Session() as sess:
    for step in range(100):
        sess.run(init)
        evals = sess.run([train, W, B])
        if step % 5 == 0:
            print(step,evals)



