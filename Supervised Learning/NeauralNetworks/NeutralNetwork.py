import tensorflow as tf

def NN_Model(data):
    I_W = tf.Variable(tf.random_normal([4,1000]))
    I_B = tf.Variable(tf.random_normal([1000]))

    I_O = tf.nn.relu((tf.matmul(data,I_W) + I_B))

    L1_W = tf.Variable(tf.random_normal([1000,500]))
    L1_B = tf.Variable(tf.random_normal([500]))

    L1_O = tf.nn.relu(tf.matmul(I_O,L1_W) + L1_B)

    O_W = tf.Variable(tf.random_normal([500, 2]))
    O_B = tf.Variable(tf.random_normal([2]))

    return tf.matmul(L1_O,O_W)+ O_B