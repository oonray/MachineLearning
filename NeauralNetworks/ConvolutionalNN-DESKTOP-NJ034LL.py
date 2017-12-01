from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from Layers.hlayer_conv import hidden_layer


#Importing The Dataset
mnist = input_data.read_data_sets('./tmp', one_hot=True)

img_size = 28*28

#number of output Layers
n_classes = 10

#the batch we get the input in
batch_size = 128


#Our Variables that will be changed later
x = tf.placeholder("float",[None,img_size])
y = tf.placeholder("float")


'''The neural network model'''
def nn_model(data):
    conv = [hidden_layer([5,5,1,32],[32]),
            hidden_layer([5,5,32,64],[64])]

    full = hidden_layer([7*7*64,1024],[1024])

    out= hidden_layer([1024],[n_classes])

    data=tf.reshape(data,shape=[-1,28,28,1])


    out.output(
        full.full(
            conv[1].maxpool(
                conv[1].convd(
                    conv[0].maxpool(
                        conv[0].convd(data))))
        )
    )


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


