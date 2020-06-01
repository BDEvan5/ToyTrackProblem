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


class ValuePolicy(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.hidden_values = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)

        self.opti = tf.keras.optimizers.RMSprop(lr=7e-3)

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        hidden_values = self.hidden_values(x)
        value = self.value(hidden_values)

        return value

class ActionPolicy(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.hidden_logs = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions)

        self.opti = tf.keras.optimizers.RMSprop(lr=7e-3)

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        hidden_logs = self.hidden_logs(x)
        logits = self.logits(hidden_logs)

        return logits


class Model:
    def __init__(self, num_actions):
        self.v_policy = ValuePolicy()
        self.a_policy = ActionPolicy(num_actions)

        self.buffer = None
        self.q_val = None
        self.update_n = 0
        self.show_plot = False

    def get_value(self, nn_state):
        value = self.v_policy.predict_on_batch(nn_state[None, :])
        value = tf.squeeze(value, axis=-1)
        value = value.numpy()[0]

        return value

    def get_action(self, nn_state):
        logits = self.a_policy.predict_on_batch(nn_state[None, :])
        # logits = tf.squeeze(logits, axis=-1)
        action = tf.random.categorical(logits, 1)
        action = tf.squeeze(action, axis=0) # axis doesn't matter (1, 1)
        action = action.numpy()[0]

        return action

    def update_model(self, buffer, show_plot=False):
        f_save = 50

        self.buffer = buffer    
        self.q_val = buffer.last_q_val
        self.show_plot = show_plot

        variables = self.v_policy.trainable_variables
        self.v_policy.opti.minimize(loss = self._loss_fcn_value, var_list=variables)

        variables = self.a_policy.trainable_variables
        self.a_policy.opti.minimize(loss=self._loss_fcn_action, var_list=variables)

        if self.update_n % f_save == 1:
            self.save_model()

        self.update_n += 1

    def _loss_fcn_value(self):
        buffer, q_val = self.buffer, self.q_val
        gamma = 0.96
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            q_vals[len(buffer)-1 - i] = q_val

        # advs = q_vals - buffer.values

        obs = tf.convert_to_tensor(buffer.states)
        values = self.v_policy(obs) 

        # value
        value_c = 0.5
        value_loss = value_c * tf.keras.losses.mean_squared_error(q_vals, values)

        if self.show_plot:
            f.plot_three(values, q_vals, value_loss)

        value_loss = tf.reduce_mean(value_loss)
        return value_loss

    def _loss_fcn_action(self):
        buffer, q_val = self.buffer, self.q_val
        gamma = 0.96
        q_vals = np.zeros((len(buffer), 1))

        for i, (_, _, _, reward, done) in enumerate(buffer.reversed()):
            q_val = reward + gamma * q_val * (1.0-done)
            q_vals[len(buffer)-1 - i] = q_val

        q_vals = q_vals[:, 0] # effectivly squeezing axis=-1
        advs = q_vals - buffer.values
        acts = np.array(buffer.actions)[:, None]

        obs = tf.convert_to_tensor(buffer.states)
        logits = self.a_policy(obs) 

        # logits
        entropy_c = 1e-4
        acts = tf.cast(acts, tf.int32)

        weighted_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_ce(acts, logits, sample_weight=advs)

        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

        logits_loss = policy_loss - entropy_c * entropy_loss

        return logits_loss

    def __call__(self, nn_state):
        value = self.get_value(nn_state)
        action = self.get_action(nn_state)

        return action, value

    def save_model(self):
        value_model = 'Networks/ValueModel'
        action_model = 'Networks/ActionModel'
        self.v_policy.save(value_model)
        self.a_policy.save(action_model)

    def load_weights(self):
        value_model = 'Networks/ValueModel'
        action_model = 'Networks/ActionModel'
        self.v_policy = tf.keras.models.load_model(value_model)
        self.a_policy = tf.keras.models.load_model(action_model)


if __name__ == "__main__":
    pass


