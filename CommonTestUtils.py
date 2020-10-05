import numpy as np 
import torch
from collections import deque
import random
import pickle

# from TestEnvWillemMod import TestEnvDQN
from matplotlib import pyplot as plt



class ReplayBufferSuper(object):
    def __init__(self, max_size=100000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.length = 0

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.length += 1
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions = [], []

        for i in ind: 
            s, a = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))

        return np.array(states), np.array(actions)

    def save_buffer(self, b_name="SuperBuf"):
        f_name = 'DataRecords/' + b_name
        b_file = open(f_name, 'wb')
        pickle.dump(self.storage, b_file)
        b_file.close() 

    def load_buffer(self, b_name="SuperBuf"):
        f_name = 'DataRecords/' + b_name
        b_file = open(f_name, 'rb')
        self.storage = pickle.load(b_file)
        


    # def save_buffer(self, b_name="SuperBuf"):
    #     f_name = 'DataRecords/' + b_name
    #     b_file = open(f_name, 'wb')
    #     pickle.dump(self.storage, b_file)
    #     b_file.close() 

    # def load_buffer(self, b_name="SuperBuf"):
    #     f_name = 'DataRecords/' + b_name
    #     b_file = open(f_name, 'rb')
    #     self.storage = pickle.load(b_file)
        


class CorridorAgent:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "Corridor"

    def act(self, obs):
        # all this does is go in the direction of the biggest range finder
        ranges = obs[2:]
        action = np.argmax(ranges)

        return action

    def load(self):
        pass



  
