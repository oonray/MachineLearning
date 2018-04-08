import gym
import os
import numpy as np
import tensorflow as tf

env = gym.make("CartPole-v1")
state = env.reset()


gamma = .9
learning_rate = 5e10
q = np.zeros(env.action_space.n)

memory = {}
memory["q"] = np.zeros([100, len(q)])
memory["state"] = np.zeros([100, len(state)])

t=0
done=False

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
try:
    for _ in range(5):
        while not done:
            if t == len(memory["q"]):
                print("[-]Memory full...")
                break
            else:
                print("-"*20)
                print("Time:",t)
                q = session.run(mod, feed_dict={x: [state]})[0]
                print("Q-Values:",q)
                memory["state"][t], memory["q"][t] = (state,q)
                action = q.argmax(axis=0)
                print("Action:",action)
                observation, reward, done, info = env.step(action)
                state = observation
                t+=1

except KeyboardInterrupt:
    pass
