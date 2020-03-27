import numpy as np 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kls
import tensorflow.keras.optimizers as klo

import random 
from collections import deque
import logging
from matplotlib import pyplot as plt 
from copy import deepcopy

batch_size = 16

class Model:
    def __init__(self, state_space, action_space):
        self.lr = 0.01
        self.epsilon = 0.99
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.gamma = 0.95

        self.action_size = action_space
        self.memory = deque(maxlen=2000)
        
        self.model = k.Sequential()
        self.model.add(kls.Dense(128, activation='relu', input_shape=(1,)))
        # self.model.add(kls.Dense(24, activation='relu'))
        self.model.add(kls.Dense(action_space, activation='linear'))
        self.model.compile(loss='mse',
                      optimizer=klo.Adam(lr=self.lr))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
class Agent:
    # basically rewriting the simulation class
    def __init__(self, state_space_n, action_space_n):
        self.model = Model(6, 3)
        self.actions_n = action_space_n
        self.state_n = state_space_n

    def train(self, env, updates, batch_size):
        # set up arrays to store data in
        ep_rewards = [0.0]

        next_obs = env.reset()
        for i in range(updates):
            for step in range(batch_size):
                obs = deepcopy(next_obs)

                action = self.model.get_action(next_obs)

                next_obs, reward, done = env.step(action)
                ep_rewards[-1] += reward

                if done:
                    ep_rewards.append(0.0) # this is the value to += for the next episode
                    plot(ep_rewards, 20)
                    next_obs = env.reset()
                    print("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

            # here write the training code to update the network

        return ep_rewards

    def test(self, env, test_number=1):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action = self.model.get_action(obs)
            obs, reward, done = env.step(action)
            ep_reward += reward
        env.sim_mem.save_ep("SimulationTests/SimulationTest%d"%test_number)
        return ep_reward


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
    print("Episode", (len(values)), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg)

def get_moving_average(period, values):
    moving_avg = 0
    if len(values) >= period:
        for i in reversed(range(period)):
            moving_avg += values[i] # adds the last 10 values
    return moving_avg

