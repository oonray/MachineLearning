import tensorflow as tf

class hidden_layer:
    def __init__(self,nodes,dataSize):
        self.n_nodes = nodes
        self.data_size = dataSize

        self.weights = tf.Variable(tf.random_normal(
                    [self.data_size,
                     self.n_nodes]
        ))

        self.biases = tf.Variable(tf.random_normal([self.n_nodes]))

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


