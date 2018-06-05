import tensorflow as tf

t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

e = .4
g = .9  #discount
a = .1#5e10 #learingRate

state_len = 4
actions = 2

s = tf.placeholder(tf.float32, [None, state_len], name='s')  # input State
s_ = tf.placeholder(tf.float32, [None, state_len], name='s_')  # input Next State
r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
action = tf.placeholder(tf.int32, [None, ], name='a')  # input Action




with tf.variable_scope('soft_replacement'):
    target_replace_op = [tf.assign(t, ft) for t, ft in zip(t_params, e_params)]

def model(x):
        with tf.variable_scope("layer1"):
            l1_w = tf.get_variable(name="l1_weights", shape=(state_len, 1000),
                                   initializer=tf.contrib.layers.xavier_initializer())
            l1_b = tf.get_variable(name="l1_biases", shape=[1000], initializer=tf.zeros_initializer())
            l1_out = tf.nn.relu(tf.matmul(x, l1_w) + l1_b, name="layer1_relu")

        with tf.variable_scope("layer2"):
            l2_w = tf.get_variable(name="l2_weights", shape=[1000, 500],
                                   initializer=tf.contrib.layers.xavier_initializer())
            l2_b = tf.get_variable(name="l2_biases", shape=[500], initializer=tf.zeros_initializer())
            l2_out = tf.nn.relu(tf.matmul(l1_out, l2_w) + l2_b, name="layer2_relu")

        with tf.variable_scope("layer3"):
            l3_w = tf.get_variable(name="l3_weights", shape=[500, 1000],
                                   initializer=tf.contrib.layers.xavier_initializer())
            l3_b = tf.get_variable(name="l3_biases", shape=[1000], initializer=tf.zeros_initializer())
            l3_out = tf.nn.relu(tf.matmul(l2_out, l3_w) + l3_b, name="layer3_relu")

        with tf.variable_scope("output"):
            o_w = tf.get_variable(name="o_weights", shape=(1000, actions), initializer=tf.contrib.layers.xavier_initializer())
            o_b = tf.get_variable(name="o_biases", shape=[actions], initializer=tf.zeros_initializer())
            return tf.matmul(l3_out, o_w) + o_b


with tf.variable_scope("eval"):
    e_mod = model(s)

with tf.variable_scope("target"):
    t_mod = model(s_)


with tf.variable_scope('q_target'):
            q_predict = r + g * tf.reduce_max(t_mod, axis=1, name='Qmax_s_')    # shape=(None, )
            q_target = tf.stop_gradient(q_predict)

with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(action)[0], dtype=tf.int32), action], axis=1)
            q_eval_wrt_a = tf.gather_nd(params=e_mod, indices=a_indices)    # shape=(None, )

with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(a).minimize(loss)

