import numpy as np 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls

import random 
from collections import deque
import logging
from matplotlib import pyplot as plt 
from copy import deepcopy
import os


class ProbDist(tf.keras.Model):
    # this is a model which takes in log probabilities and returns an action by sampling methods
    def call(self, logits, **kwags):
        print("Logits: " + str(logits))
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, actions_n):
        super().__init__('mlp_policy') # not sure what that is

        self.hidden_values = kl.Dense(128, activation='relu', input_shape=(1,))
        self.values = kl.Dense(1, name='values')

        self.hidden_actor = kl.Dense(128, activation='relu', input_shape=(1,))
        self.logits = kl.Dense(actions_n, name='policy_logits')

        self.dist = ProbDist()

    def call(self, inputs, **kwags):
        x = tf.convert_to_tensor(inputs)
        # print("Tensor x in call")
        # print(x)

        hidden_logs = self.hidden_actor(x)
        logits = self.logits(hidden_logs)

        hidden_values = self.hidden_values(x)
        values = self.values(hidden_values)

        return logits, values

    def action_value(self, obs):
        # this apparently executes call

        logits, value = self.predict_on_batch(obs)
        # print("ActionValueLogits: " + str(logits))
        action = self.dist.predict_on_batch(logits)

        action_ret = np.squeeze(action, axis=-1)
        value_ret = np.squeeze(value, axis=-1)

        return action_ret, value_ret

    
class Agent:
    # basically rewriting the simulation class
    def __init__(self, state_space_n, action_space_n):
        self.gamma = 0.99
        self.value_c = 0.5
        self.entropy_c = 1e-4
        lr = 7e-3
        self.update_network_const = 10
        self.weigth_path = "ModelWeights/target_weights"

        self.model = Model(action_space_n)
        self.model.compile(
        optimizer=ko.RMSprop(lr=lr),
        loss=[self._logits_loss, self._value_loss])

        # obs0 = np.zeros(12)[None, :]
        # a, v = self.model.action_value(obs0)
    
        # self.model.summary()

        self.target = Model(action_space_n)
        self.target.compile(
        optimizer=ko.RMSprop(lr=lr),
        loss=[self._logits_loss, self._value_loss])

        self.actions_n = action_space_n
        self.state_n = state_space_n

        for file_name in os.listdir(os.getcwd() + "/SimulationTests/"):
            os.remove("SimulationTests/" + file_name) # deletes old files
            print("File deleted: " + str(file_name))

    def train(self, env, updates, batch_size):
        actions = np.empty((batch_size), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + (self.state_n,))

        ep_rewards = [0.0]
        next_obs = env.reset()
        # print(next_obs)
        re_flag = 0
        i = 0 # for update to target network
        for update in range(updates):
            i += 1
            for step in range(batch_size):
                next_obs = next_obs[None, :]
                observations[step] = deepcopy(next_obs)
                actions[step], values[step] = self.model.action_value(next_obs)

                next_obs, rewards[step], dones[step] = env.step(actions[step])
                # print(next_obs)
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    plot(ep_rewards, 10)
                    # print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # train my target and predict on batch
            acts_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            losses = self.target.train_on_batch(observations, [acts_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

            if i % self.update_network_const == 0:
                print("Updating model weights")
                self.target.save_weights(self.weigth_path)
                self.model.load_weights(self.weigth_path)

        return ep_rewards

    def test(self, env, test_number=1):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done = env.step(action)
            ep_reward += reward

        env.sim_mem.save_ep("SimulationTests/SimulationTest%d"%test_number)
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # this is a new array to store the returns in, returns are the rewards that have been adjusted. 

        for t in reversed(range(rewards.shape[0])):
            # this is the first part of del
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1-dones[t])
        returns = returns[:-1]

        advantages = returns - values # this is the del
        return returns, advantages

    def _value_loss(self, returns, value):
        loss = self.value_c * kls.mean_squared_error(returns, value)

        return loss

    def _logits_loss(self, actions_advantages, log_probs):
        actions, advs = tf.split(actions_advantages, 2, axis=-1)

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, log_probs, sample_weight=advs)

        probs = tf.nn.softmax(log_probs)

        entropy_loss = kls.categorical_crossentropy(probs, probs)

        loss = policy_loss - self.entropy_c * entropy_loss

        return loss



def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    # plt.plot(moving_avg)    
    plt.pause(0.001)


def get_moving_average(period, values):
    moving_avg = 0
    if len(values) >= period:
        for i in reversed(range(period)):
            moving_avg += values[i] # adds the last 10 values
    return moving_avg


