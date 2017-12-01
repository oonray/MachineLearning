from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from .Layers.recurrent_h_layer import hidden_layer

mnist = input_data.read_data_sets('./tmp', one_hot=True)

epocs = 10
n_classes = 10

batch_size,rnn_size = [128,128]

chunk_size,n_chunks  = 28,28

x = tf.placeholder("float",[None, n_chunks,chunk_size])
y = tf.placeholder("float")

def rec_nn_model(x):
    layer = hidden_layer(rnn_size,n_classes)

    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x,n_chunks,axis=0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    output, state = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)

    output = layer.output(output[-1])

    return output

def rec_nn_train(inpt):
    result = rec_nn_model(inpt)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,labels=y))
    optimiser = tf.train.AdamOptimizer().minimize(cost)



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for ep in range(epocs):
            e_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x, e_y = mnist.train.next_batch(batch_size)

                e_x = e_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimiser,cost],feed_dict={x: e_x,y: e_y})
                e_loss += c

            print("Epoch {} done. Loss {}".format(ep,e_loss))

        corr = tf.equal(tf.argmax(result,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(corr,"float"))

        print("Acccuracy {}".format(accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)),y:mnist.test.labels})))



rec_nn_train(x)


