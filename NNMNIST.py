from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from classes.hidden_layer import hidden_layer

mnist = input_data.read_data_sets('./tmp', one_hot=True)

img_size = 28*28

n_nodes_hidden_layer1 = 500
n_nodes_hidden_layer2 = 500
n_nodes_hidden_layer3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder("float",[None,img_size])
y = tf.placeholder("float")

def nn_model(data):
    layer1 = hidden_layer(n_nodes_hidden_layer1,img_size)
    layer2 = hidden_layer(n_nodes_hidden_layer2,n_nodes_hidden_layer1)
    layer3 = hidden_layer(n_nodes_hidden_layer3,n_nodes_hidden_layer2)
    output = hidden_layer(n_classes,n_nodes_hidden_layer3)

    return output.output(
        layer3.nn(
            layer2.nn(
                layer1.nn(data))))


def nn_train(inpt):
    result = nn_model(inpt)
    tf.reduce_mean(tf.nn.softmax(result,y))







