import numpy as np
import tensorflow as tf
import gym
from matplotlib import pyplot as plt
from PathTracker import Tracker
import LibFunctions as f


class BufferVanilla():
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []

        self.last_q_val = None

    def add(self, state, action, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.states, 
                self.actions,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

    def print_rewards(self):
        print(self.rewards)

    def print_batch(self):
        new_values = []
        for val in self.values:
            new_values.append(val.numpy()[0])
        zipped = zip(new_values, self.rewards, self.dones)
        for iteration in zipped:
            print(iteration)


class ReplayBuffer:
    def __init__(self, size=5000):
        self.size = 5000
        self.buffer = []
        self.idx = 0

    def add_batch(self, batch):
        if self.idx > self.size:
            self.buffer.pop(0) # remove oldest batch
        self.buffer.append(batch)
        self.idx += 1

    def get_random_batch(self):
        # each time this is called, it will get a random buffer for training.
        rand_idx = np.random.randint(0, self.idx)
        buffer_return = self.buffer[rand_idx]

        return buffer_return


def plot(values, moving_avg_period=10, title="Training", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 5, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def get_moving_average(period, values):

    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg


class Policy(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.hidden_values = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)

        self.opti = tf.keras.optimizers.RMSprop(lr=7e-3)

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        hidden_values = self.hidden_values(x)
        value = self.value(hidden_values)

        return value


class Model:
    def __init__(self, num_actions):
        self.policy = Policy(num_actions)

        self.buffer = None
        self.q_val = None
        self.update_n = 0

    def get_action_value(self, obs):
        value = self.policy.predict_on_batch(obs)
        value = tf.squeeze(value, axis=-1)

        return value

    def update_model(self, buffer):
        self.buffer = buffer
        self.q_val = buffer.last_q_val

        variables = self.policy.trainable_variables
        self.policy.opti.minimize(loss = self._loss_fcn, var_list=variables)

        self.update_n += 1

    def _loss_fcn(self):
        buffer, q_val = self.buffer, self.q_val
        gamma = 0.96
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            q_vals[len(buffer)-1 - i] = q_val

        # advs = q_vals - buffer.values

        obs = tf.convert_to_tensor(buffer.states)
        values = self.policy(obs) 

        # value
        value_c = 0.5
        value_loss = value_c * tf.keras.losses.mean_squared_error(q_vals, values)

        # f.plot_three(values, q_vals, value_loss)

        value_loss = tf.reduce_mean(value_loss)

        total_loss = value_loss #+ logits_loss
        return total_loss

    def __call__(self, nn_state):
        value = self.policy.predict_on_batch(nn_state[None, :])
        value = tf.squeeze(value, axis=-1)
        value = value.numpy()[0]

        return value




if __name__ == "__main__":
    learn()


