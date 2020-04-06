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
    
class Agent_A2C:
    # basically rewriting the simulation class
    def __init__(self, state_space_n, action_space_n):
        self.gamma = 0.99
        self.value_c = 0.5
        self.entropy_c = 1e-4
        lr = 7e-3
        self.update_network_const = 10
        self.weight_path = "ModelWeights/target_weights"

        self.model = Model(action_space_n)
        self.model.compile(
        optimizer=ko.RMSprop(lr=lr),
        loss=[self._logits_loss, self._value_loss])

        self.actions_n = action_space_n
        self.state_n = state_space_n

        self.agent_file_path = "Agent_A2C_SimTests/"
        self.agent_test_path = "Agent_A2C_SimTests/AgentTests/"
        self.weight_path = self.agent_file_path + "ModelWeights/target_weights"

    def clear_test_files(self):
        file_path_list = ["EpHistories/", "Plots/", "TrainingImages/"]
        for path in file_path_list:
            file_path = self.agent_file_path + path
            for file_name in os.listdir(os.getcwd() + "/" + file_path):
                os.remove(file_path + file_name) # deletes old files
                print("File deleted: " + str(file_path) + "/" + str(file_name))

    def train(self, env, steps, train_name="TrainName", f_test=20):
        print(train_name)
        batch_size = 32
        updates = int(steps / batch_size)

        # set up arrays
        ep_rewards = [0.0]
        actions = np.empty((batch_size), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + (self.state_n,))

        next_obs = env.reset()

        for update in range(updates):
            for step in range(batch_size):
                observations[step] = deepcopy(next_obs)
                actions[step], values[step] = self.model.action_value(next_obs)

                next_obs, rewards[step], dones[step] = env.step(actions[step])
                # print(next_obs)
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    plot(ep_rewards, 10, train_name)

                    if len(ep_rewards) % f_test == 1: # test and save every few 
                        self.test(env, train_name + str(len(ep_rewards)))
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs)
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)

            # train my target and predict on batch
            acts_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            losses = self.model.train_on_batch(observations, [acts_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        self.model.save_weights(self.agent_file_path + "ModelWeights/target_weights")
        self.test(env, train_name + ":FinalTest")
        plt.figure(2)
        plot_path = self.agent_file_path + "Plots/" + "Plot --> " + train_name + ".png"
        print(plot_path)
        plt.savefig(plot_path)
        return ep_rewards

    def test(self, env, test_name="TestName"):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs)
            obs, reward, done = env.step(action)
            ep_reward += reward
        env.render_episode(self.agent_test_path + test_name)
        print("A2C Agent tested, ep reward = %d --> "%ep_reward + test_name)
        env.sim_mem.save_ep(self.agent_test_path + test_name)
        return ep_reward

    def load_weights(self):
        self.model.load_weights(self.agent_file_path + "ModelWeights/target_weights")

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


def plot(values, moving_avg_period, title):
    plt.figure(2)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 2, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    # print("Episode", (len(values)), "\n", \
    #     moving_avg_period, "episode moving avg:", moving_avg)

def get_moving_average(period, values):
    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg
