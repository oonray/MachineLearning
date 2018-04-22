import gym
import inspect
import numpy as np
import tensorflow as tf
import pandas as pd

env = gym.make("CartPole-v1")
state = env.reset()

e = .9
gamma = .9 #discount
alpha = 5e10 #learingRate

Q = pd.DataFrame(np.zeros((1,env.action_space.n)), columns=range(env.action_space.n))
print(Q)


"""
x = tf.placeholder("float")
y = tf.placeholder("float")

def model(x):
    x = tf.cast(x, tf.float32)
    w1 = tf.Variable(tf.random_normal([len(state), 1000]))
    b1 = tf.Variable(tf.random_normal([1000]))
    o1 = tf.nn.relu((tf.matmul(x, w1)+b1))

    w2 = tf.Variable(tf.random_normal([1000, len(q)]))
    b2 = tf.Variable(tf.random_normal([len(q)]))
    return tf.matmul(o1,w2) + b2

session = tf.Session()

mod = model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mod, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print("[+]Init.....")
session.run(tf.global_variables_initializer())

print("[+]Starting.....")
"""
