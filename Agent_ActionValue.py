import numpy as np 

import random 
from collections import deque
import logging
from matplotlib import pyplot as plt 
from copy import deepcopy
import os

    
class Agent_ActionValue:
    def __init__(self, config, model_net, buffer, env):
        self.config = config
        self.model = model_net
        self.buffer = buffer
        self.env = env

    def run_sim(self):
        next_obs, done, ep_reward = self.env.reset(), False, 0
        while not done:
            obs = deepcopy(next_obs)
            action = self.choose_action(obs)
            next_obs, reward, done = self.env.step(action)
            ep_reward += reward
            self.buffer.save_step((obs, action, reward, next_obs, done))
        return ep_reward

    def run_test(self):
        next_obs, done, ep_reward = self.env.reset(), False, 0
        while not done:
            obs = deepcopy(next_obs)
            action = self.model.get_action(obs)
            next_obs, reward, done = self.env.step(action)
            ep_reward += reward
            self.buffer.save_step((obs, action, reward, next_obs, done))
        return ep_reward

    def choose_action(self, state):
        if np.random.rand() <= self.config.eps:
            return random.randint(0, self.config.action_space-1)
        else:
            return self.model.get_action(state)


class Trainer_AV:
    def __init__(self, config, target_net):
        self.config = config
        self.target = target_net

    def train_network(self, batch):
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target_prediction = self.target.predict(next_state)
                target = (reward + self.config.gamma *
                          np.amax(target_prediction[0]))
                # print(target)
            target_f = self.target.predict(state)
            target_f[0][action] = target
            self.target.fit(state, target_f, epochs=1, verbose=0)

        self.config.step_eps()

    def test_network(self, batch):
        # lists to store the data in
        x = []
        y = []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target_prediction = self.target.predict(next_state)
                target = (reward + self.config.gamma *
                          np.amax(target_prediction[0]))

            target_f = self.target.predict(state)
            target_f[0][action] = target

            x.append(state[:][0])
            y.append(target_f)

        x = np.array(x)
        y = np.array(y)


        loss = self.target.test_on_batch(x, y)
        mean_loss = np.mean(loss)
        print(mean_loss)
        return mean_loss



