import tensorflow as tf

class hidden_layer:
    def __init__(self,w,b):
        self.w = tf.Variable(tf.random_normal(w))
        self.b = tf.Variable(tf.random_normal(b))

    def convd(self,x,W):
        return tf.nn.relu(tf.nn.conv2d(x,W+self.b,strides=[1,1,1,1],padding="SAME"))

    def maxpool(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    def full(self,x):
        return tf.nn.relu(tf.matmul(tf.reshape(x,[-1, 7*7*64]), self.w) + self.b)

    def output(self,x):
        return tf.matmul(x, self.w) + self.b


