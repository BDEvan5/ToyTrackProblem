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
        return self.env.sim_mem, ep_reward

    def choose_action(self, state):
        if np.random.rand() <= self.config.eps:
            return random.randint(0, self.config.action_space-1)
        else:
            return self.model.get_action(state)


class Trainer_AV:
    def __init__(self, config, target_net):
        self.config = config
        self.target = target_net

    def train_network(self, batch, epochs=1):
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target_prediction = self.target.predict(next_state)
                target = (reward + self.config.gamma *
                          np.amax(target_prediction[0]))
                # print(target)
            target_f = self.target.predict(state)
            target_f[0][action] = target
            self.target.fit(state, target_f, epochs=epochs, verbose=0)

        self.config.step_eps()




