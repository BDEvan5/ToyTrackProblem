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
    map_name = 'TestTrack1000'
    env = TestEnv(map_name)

    score, done, state = 0, False, env.reset()
    while not done:
        a = agent.act(state)
        s_prime, _, done, _ = env.step(a)
        state = s_prime
        score += 1 # counts number of steps
        if show:
            env.box_render()
            # env.full_render()
            pass
        
    print(f"SingleRun --> Score: {score} --> steps: {env.steps}")
    return score
