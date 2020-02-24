import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kls
import tensorflow.keras.optimizers as klo

import numpy as np 
import random
import gym
from collections import deque

import logging
import EpisodeMem

class DQN:
    # this class is going to be the model and be accessed from the main function
    # this is like a memory test to copy the other example.
    def __init__(self, logger, state_space, action_space):
        self.logger = logger

        self.lr = 0.3
        self.epsilon = 0.99
        self.epsilon_decay = 0.99995
        self.min_epsilon = 0.01
        self.gamma = 0.95

        self.action_size = action_space
        self.memory = deque(maxlen=2000)
        
        self.model = k.Sequential()
        self.model.add(kls.Dense(state_space, activation='relu'))
        self.model.add(kls.Dense(24, activation='relu'))
        self.model.add(kls.Dense(24, activation='relu'))
        self.model.add(kls.Dense(action_space))
        self.model.compile(loss='mse',
                      optimizer=klo.Adam(lr=self.lr))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

#TODO: understand this
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            msg = "State: " + str(state) + " Action: " + str(action) + " Reward: " + str(reward) + " Next state:" + str(next_state)
            self.logger.debug(msg)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

class RL_Controller:
    def __init__(self, env, logger, batch_size=8):
        self.env = env
        self.batch_size = batch_size
        self.agent = DQN(logger, self.env.state_space, self.env.action_space)

        self.state = None
        self.obs = None
        self.action = None

        self.logger = logger
        self.ep = EpisodeMem.EpisodeMem()
        
    def run_learning(self, max_time=50, episodes=1):

        for e in range(episodes):
            obs = self.env.reset()
            self.state = self.morph_state(obs)
            cum_reward = 0
            for t in range(max_time):
                
                network_action = self.agent.get_action(self.state)

                self.action = self.morph_action(network_action)
                msg = "Action: " + str(self.action)

                # print(msg)                

                new_obs, reward, done = self.env.step(self.action)
                cum_reward += reward
                
                new_state = self.morph_state(new_obs)
                self.agent.memorize(self.state, network_action, reward, new_state, done)
                self.state = new_state

                self.ep.add_step(new_obs, self.action, reward)

                if t == max_time-1:
                    print("Max Steps reached")
                    done = True

                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                        .format(e, episodes, cum_reward, self.agent.epsilon))
                    break

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

    def run_test_learning(self, max_time=250, episodes=1):

        for e in range(episodes):
            obs = self.env.reset()
            self.state = self.morph_state(obs)
            cum_reward = 0
            for t in range(max_time):
                self.env.render()
                
                network_action = self.agent.get_action(self.state)

                self.action = self.morph_action(network_action)
                msg = "Action: " + str(self.action)

                # print(msg)                

                new_obs, reward, done = self.env.step(self.action)
                cum_reward += reward
                
                new_state = self.morph_state(new_obs)
                self.agent.memorize(self.state, network_action, reward, new_state, done)
                self.state = new_state

                if t == max_time-1:
                    print("Max Steps reached")
                    done = True

                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                        .format(e, episodes, cum_reward, self.agent.epsilon))
                    break

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

    def morph_state(self, obs):
        ret_state = np.zeros((2, 1))
        for i in range(2):
            ret_state[i] = obs.x[i]
        # for i in range(9):
        #     ret_state[i+2] = obs.senses[i].val
        
        return ret_state

    def morph_action(self, network_action):
        action = [0, 0]
        scale = 10
        action[0] = network_action / 3 -1
        action[1] = (network_action-1) % 3 -1
        for i in range(2):
            action[i] *= scale
        return action 

    def set_locations(self, x_start, x_end):
        self.env.add_locations(x_start, x_end)

    def get_ep_mem(self):
        return self.ep

