import numpy as np 
import torch
from collections import deque
import random
import pickle

# from TestEnvWillemMod import TestEnvDQN
from matplotlib import pyplot as plt


MEMORY_SIZE = 100000

class ReplayBufferDQN():
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_SIZE)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def memory_sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # todo: move the tensor bit to the agent file, just return lists for the moment.
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

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
        
class ReplayBufferAuto(object):
    def __init__(self, max_size=1000000):     
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)


    def save_buffer(self, b_name="SuperBuf"):
        f_name = 'DataRecords/' + b_name
        b_file = open(f_name, 'wb')
        pickle.dump(self.storage, b_file)
        b_file.close() 

    def load_buffer(self, b_name="SuperBuf"):
        f_name = 'DataRecords/' + b_name
        b_file = open(f_name, 'rb')
        self.storage = pickle.load(b_file)
        


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



  
