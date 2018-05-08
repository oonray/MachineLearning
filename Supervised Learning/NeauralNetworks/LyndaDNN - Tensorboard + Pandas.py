import pandas as pd
import tensorflow as tf

training_data = pd.read_csv("./Ex_Files_TensorFlow/sales_data_training.csv")
test_data = pd.read_csv("./Ex_Files_TensorFlow/sales_data_test.csv")

data_len = 9

with tf.VariableScope("Input"):
    X = tf.placeholder(tf.float32,name="X")


def get_model(inn):
    with tf.VariableScope("layer1"):
        l1_w = tf.get_variable(name="l1_weights",shape=[None, data_len],initializer=tf.random_normal_initializer)
        l1_b = tf.get_variable(name="l1_biases",shape=[100])
        l1_out = tf.nn.relu(tf.add(l1_w*inn, name="layer1_add")+l1_b,name="layer1_relu")

    with tf.VariableScope("layer2"):
        l2_w = tf.get_variable(name="l2_weights",shape=[100, 50],initializer=tf.random_normal_initializer)
        l2_b = tf.get_variable(name="l2_biases",shape=[50])
        l2_out = tf.nn.relu(tf.add(l2_w*l1_out)+l2_b, name="layer2_relu")

    with tf.VariableScope("layer3"):
        l3_w = tf.get_variable(name="l3_weights", shape=[50, 100], initializer=tf.random_normal_initializer)
        l3_b = tf.get_variable(name="l3_biases", shape=[100])
        l3_out = tf.nn.relu(   tf.add(l3_w * l2_out) + l3_b, name="layer3_relu")

    with tf.VariableScope("output"):
        o_w = tf.get_variable(name="o_weights",shape=(100,1),initializer=tf.random_normal_initializer)
        o_b = tf.get_variable(name="o_biases",shape=[1])
        return tf.add(o_w,l3_out,name="o_add")+o_b


with tf.VariableScope("predict"):
    model = get_model(X)

with tf.VariableScope("cost"):
    Y = tf.placeholder(tf.float32, name="Y", shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(model, Y, name="Squared Difference"),name="Mean")

with tf.VariableScope("train"):
    optimizer = tf.train.AdamOptimizer().minimize(cost)


with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   cost = sess.run(optimizer, feed_dict={X:training_data.drop("total_earnings"), Y:training_data["total_earnings"]})
