from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

state_height = 105

# Width of each image-frame in the state.
state_width = 80

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])

# Number of images in the state.
state_channels = 2

# Shape of the state-array.
state_shape = [state_height, state_width, state_channels]


class NeuralNetwork:
    def __init__(self, num_actions, replay_memory, state_len, restore=False):
        self.file = "./file.bak"
        self.session = tf.Session()

        self.memory = replay_memory

        self.num_actions = num_actions
        self.state_len = state_len

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, state_len], name='x')

        self.learning_rate = 1e-3
        self.y= tf.placeholder(tf.float32,shape=[None, num_actions], name='y')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.nn_model(self.x), labels=self.y))
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.saver = tf.train.Saver()

    def nn_model(self,data):
        data = tf.reshape(data, shape=[-1,self.state_len])

        CW = tf.Variable(tf.random_normal([self.state_len,32]))
        CB = tf.Variable(tf.random_normal([32]))

        one = tf.nn.relu((tf.matmul(data,CW)+CB))

        C2W = tf.Variable(tf.random_normal([32,64]))
        C2B = tf.Variable(tf.random_normal([64]))

        two = tf.nn.relu((tf.matmul(one,C2W)+C2B))

        FULLW = tf.Variable(tf.random_normal([64,1024]))
        FULLB = tf.Variable(tf.random_normal([1024]))

        full = tf.nn.relu((tf.matmul(two,FULLW)+FULLB))

        OUTW = tf.Variable(tf.random_normal([1024,self.num_actions]))
        OUTB = tf.Variable(tf.random_normal([self.num_actions]))

        return tf.add(tf.matmul(full,OUTW),OUTB)

    def close(self):
        self.session.close()

    def save(self):
        self.saver.save(self.session,self.file)

    def load(self):
        self.saver.restore(self.session,self.file)

    def getQ(self,inp):
        return self.session.run([self.optimiser,self.cost],feed_dict={self.x:inp})

    def Train(self, min_epochs=1.0, max_epochs=10,
                 batch_size=128, loss_limit=0.015,
                 learning_rate=1e-3):

        self.memory.prepare_sampling_prob(batch_size=batch_size)

        iterations_per_epoch = self.memory.num_used / batch_size

        min_iterations = int(iterations_per_epoch * min_epochs)
        max_iterations = int(iterations_per_epoch * max_epochs)
        loss_history = np.zeros(100, dtype=float)

        for i in range(max_iterations):
            state_batch, q_values_batch = self.memory.random_batch()

            feed_dict = {self.x: state_batch,
                         self.y: q_values_batch,
                         self.learning_rate: learning_rate}

            loss_val, _ = self.session.run([self.cost, self.optimiser],
                                           feed_dict=feed_dict)

            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val
            loss_mean = np.mean(loss_history)

            if i > min_iterations and loss_mean < loss_limit:
                break




