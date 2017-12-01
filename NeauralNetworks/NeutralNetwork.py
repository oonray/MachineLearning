from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from .Layers.hidden_layer import hidden_layer


#Importing The Dataset
mnist = input_data.read_data_sets('./tmp', one_hot=True)

img_size = 28*28

#Hidden Layer Sizes
n_nodes_hidden_layer1 = 500
n_nodes_hidden_layer2 = 500
n_nodes_hidden_layer3 = 500

#number of output Layers
n_classes = 10

#the batch we get the input in
batch_size = 100


#Our Variables that will be changed later
x = tf.placeholder("float",[None,img_size])
y = tf.placeholder("float")


'''The neural network model'''
def nn_model(data):
    '''layers can be scaled at will
    adding a layer requires adding the input of the last layer as inp data
    '''

    '''layer1'''
    layer1 = hidden_layer(n_nodes_hidden_layer1,img_size)

    '''layer2'''
    layer2 = hidden_layer(n_nodes_hidden_layer2,n_nodes_hidden_layer1)

    '''layer3'''
    layer3 = hidden_layer(n_nodes_hidden_layer3,n_nodes_hidden_layer2)

    '''output layer'''
    output = hidden_layer(n_classes,n_nodes_hidden_layer3)

    '''Running network using class hidden_layer.py
    Each layer gets the preceeding layers ouptut as input'''
    return output.output(
        layer3.nn(
            layer2.nn(
                layer1.nn(data))))


'''The function that trains our model'''
def nn_train(inpt):
    """NN MODEL that we are running"""
    result = nn_model(inpt)

    """The cost function"""
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,labels=y))

    """The variabe optimizer"""
    optimiser = tf.train.AdamOptimizer().minimize(cost)

    """Number of epocs"""
    epocs = 10

    """Starting the tensorflow Session"""
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for ep in range(epocs):
            e_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x, e_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimiser,cost],feed_dict={x: e_x,y: e_y})
                e_loss += c

            print("Epoch {} done. Loss {}".format(ep,e_loss))

        corr = tf.equal(tf.argmax(result,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(corr,"float"))

        print("Acccuracy {}".format(accuracy.eval({x:mnist.test.images,y:mnist.test.labels})))



nn_train(x)


