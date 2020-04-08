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


class AgentA2C:
    def __init__(self, config, model_net, buffer, env):
        self.config = config
        self.model = model_net
        self.buffer = buffer
        self.env = env

    def run_sim(self):
        next_obs, done, ep_reward = self.env.reset(), False, 0
        while not done:
            obs = deepcopy(next_obs)
            action, value = self.model.action_value(obs)
            action = self.choose_action(obs)
            next_obs, reward, done = self.env.step(action)
            ep_reward += reward
            self.buffer.save_step((obs, action, value, reward, next_obs, done))
        return ep_reward


class TrainerA2C:
    def __init__(self, config, target_net):
        self.config = config
        self.target = target_net

    def train_network(self, batch):
        states = batch[:, 0]
        actions = batch[:, 1]
        values = batch[:, 2]
        rewards = batch[:, 3]
        next_states = batch[:, 4]
        dones = batch[:, 5]
        next_values = np.zeros_like(next_states)

        # generates next values for each state
        for i, next_state in enumerate(next_states):
            _, value = self.target.action_value(next_state)
            next_values[i] = value
            
        returns, advs = self._returns_advantages(rewards, dones, values, next_values)

    def _returns_advantages(self, rewards, dones, values, next_values):
        returns = np.append(np.zeros_like(rewards), next_values, axis=-1)
        # this is a new array to store the returns in, returns are the rewards that have been adjusted. 

        for t in reversed(range(rewards.shape[0])):
            # this is the first part of del
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1-dones[t])
        returns = returns[:-1]

        advantages = returns - values # this is the del
        return returns, advantages


class Agent_A2C:
    # basically rewriting the simulation class
    def __init__(self, state_space_n, action_space_n):




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
