import numpy as np
import tensorflow as tf

state_height = 105

# Width of each image-frame in the state.
state_width = 80

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])

# Number of images in the state.
state_channels = 2

# Shape of the state-array.
state_shape = [state_height, state_width, state_channels]

class NeuralNetwork:
    def __init__(self,num_actions,replay_memory):

        self.dir = "./Ai.back"
        self.actions = num_actions
        self.replay_memory = replay_memory
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,self.actions], name='y')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, num_actions],
                                           name='q_values_new')

        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')

        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')

        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)

        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        self.model = self.get_model(self.x)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.y))
        self.optimiser = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

        self.saver = tf.train.Saver()

        self.session = tf.Session()

    def get_model(self,data):
        self.weights = [
            tf.Variable(tf.random_normal([3,3,1,16])),
            tf.Variable(tf.random_normal([3,3,16,32])),
            tf.Variable(tf.random_normal([3,3,43,64])),
            tf.Variable(tf.random_normal([state_height * state_width * 64,1024])),
            tf.Variable(tf.random_normal([state_height * state_width * 64, 1024])),
            tf.Variable(tf.random_normal([state_height * state_width * 64, 1024])),
            tf.Variable(tf.random_normal([state_height * state_width * 64, 1024])),
            tf.Variable(tf.random_normal([1024, self.actions])),
                   ]

        self.biases = [
            tf.Variable(tf.random_normal([3, 3, 1, 16])),
            tf.Variable(tf.random_normal([3, 3, 16, 32])),
            tf.Variable(tf.random_normal([3, 3, 43, 64])),
            tf.Variable(tf.random_normal([1024])),
            tf.Variable(tf.random_normal([1024])),
            tf.Variable(tf.random_normal([1024])),
            tf.Variable(tf.random_normal([1024])),
            tf.Variable(tf.random_normal([1024]))
        ]

        self.layers = [
            tf.nn.relu(tf.nn.conv2d(data, tf.add(self.weights[0], self.biases[0]), strides=[1, 2, 2, 1], padding="SAME")),
            tf.nn.max_pool(self.layers[0], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"),

            tf.nn.relu(tf.nn.conv2d(self.layers[1], tf.add(self.weights[1], self.biases[1]), strides=[1, 2, 2, 1], padding="SAME")),
            tf.nn.max_pool(self.layers[2], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"),

            tf.nn.relu(tf.nn.conv2d(self.layers[3], tf.add(self.weights[2], self.biases[2]), strides=[1, 1, 1, 1],padding="SAME")),
            tf.nn.max_pool(self.layers[4], ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME"),

            tf.nn.relu(tf.add(tf.matmul(tf.reshape(self.layers[5], [-1, state_height * state_width * 64]), self.weights[3]), self.biases[3])),
            tf.nn.relu(tf.add(tf.matmul(tf.reshape(self.layers[6], [-1, state_height * state_width * 64]), self.weights[4]), self.biases[4])),
            tf.nn.relu(tf.add(tf.matmul(tf.reshape(self.layers[7], [-1, state_height * state_width * 64]), self.weights[5]), self.biases[5])),
            tf.nn.relu(tf.add(tf.matmul(tf.reshape(self.layers[8], [-1, state_height * state_width * 64]), self.weights[6]), self.biases[6])),
        ]

        return tf.add(tf.matmul(self.layers[-1], self.layers[-1]), self.biases[-1])
    def close(self):
        self.session.clode()
    def load_checkpoint(self):
        try:
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.dir)
            self.saver.restore(self.session, save_path=last_chk_path)
        except:
            self.session.run(tf.global_variables_initializer())
    def save_checkpoint(self, current_iteration):
        self.saver.save(self.session,
                        save_path=self.dir,
                        global_step=current_iteration)



