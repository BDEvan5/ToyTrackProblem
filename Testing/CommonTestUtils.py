import numpy as np 
import torch
from collections import deque
import random
from TestEnv import TestEnv


MEMORY_SIZE = 100000

class ReplayBuffer():
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
# todo: move tensors to the agent.



"""Single Evals"""
def single_rep_eval(agent, show=True):
    env = TestEnv()
    env.map_1000()
    # env.map_1010()
    # env.map_1020()

    score, done, state = 0, False, env.reset()
    while not done:
        a = agent.act(state)
        s_prime, _, done, _ = env.step(a)
        state = s_prime
        score += 1 # counts number of steps
        if show:
            env.box_render()
            env.full_render()
            pass
        
    print(f"SingleRun --> Score: {score} --> Steps: {env.steps}")
    return score


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



class PurePursuit:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "PurePursuit"

    def load(self):
        pass

    def act(self, obs):
        try:
            grad = obs[1] / obs[0] # y/x
        except:
            grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.act_space - 1)
        action = int(angle / dth)

        return action
