import numpy as np

state_height = 105

# Width of each image-frame in the state.
state_width = 80

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])

# Number of images in the state.
state_channels = 2

# Shape of the state-array.
state_shape = [state_height, state_width, state_channels]


class ReplayMemory:
    def __init__(self, size, num_actions, discount_factor=0.97):
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)


        self.actions = np.zeros(shape=size, dtype=np.int)
        self.rewards = np.zeros(shape=size, dtype=np.float)


        self.end_life = np.zeros(shape=size, dtype=np.bool)
        self.end_episode = np.zeros(shape=size, dtype=np.bool)
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        self.size = size

        self.discount_factor = discount_factor

        self.num_used = 0

        self.error_threshold = 0.1

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode):

        if not self.is_full():
            k = self.num_used

            self.num_used += 1

            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode

            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update_all_q_values(self):

        self.q_values_old[:] = self.q_values[:]

        for k in reversed(range(self.num_used - 1)):
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]
            end_episode = self.end_episode[k]

            if end_life or end_episode:
                action_value = reward
            else:
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            self.q_values[k, action] = action_value

        self.print_statistics()

    def prepare_sampling_prob(self, batch_size=128):
        err = self.estimation_errors[0:self.num_used]

        idx = err < self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))


        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        idx = np.concatenate((idx_lo, idx_hi))

        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        begin = 0
        while begin < self.num_used:
            end = begin + batch_size
            if end > self.num_used:
                end = self.num_used
            progress = end / self.num_used
            yield begin, end, progress
            begin = end

    def estimate_all_q_values(self, model):
        for begin, end, progress in self.all_batches():
            states = self.states[begin:end]
            self.q_values[begin:end] = model.get_q_values(states=states)



