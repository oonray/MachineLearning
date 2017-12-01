import tensorflow as tf

class hidden_layer:
    def __init__(self,rnn_size,classes):
        self.n_nodes = rnn_size
        self.data_size = classes

        self.weights = tf.Variable(tf.random_normal([self.n_nodes,self.data_size]))
        self.biases = tf.Variable(tf.random_normal([self.data_size]))

    def nn(self,data):
        return tf.nn.relu(tf.matmul(data, self.weights) + self.biases)

    def output(self,data):
        return tf.matmul(data, self.weights) + self.biases

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


