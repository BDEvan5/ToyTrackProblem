import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import collections
import random
import gym
import numpy as np 
from matplotlib import pyplot as plt

from SimpleEnv import MakeEnv
import LibFunctions as lib 



#Hyperparameters
learning_rate = 0.005
gamma         = 0.98
buffer_limit  = 10000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)

        return h3

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        return self.forward(obs).argmax().item()

class AgentDQN:
    def __init__(self, state_size, action_size):
        self.q = Qnet(state_size, action_size)
        self.q_target = Qnet(state_size, action_size)
        self.q.load_state_dict(self.q_target.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate) 
        self.memory = ReplayBuffer()

        self.n = 0 # count steps
        self.eps = 0.08

    def train(self):
        for i in range(10):
            s, a, r, sp_prime, done = self.memory.sample(batch_size)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(sp_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        epsilon = max(0.01, 0.08 - 0.01*(self.n/2000)) #Linear annealing from 8% to 1%
        self.eps = epsilon
        self.n += 1
        a = self.q.sample_action(state, epsilon)

        return a

    def add_mem_step(self, transition):
        self.memory.put(transition)

    def update_parameters(self):
        self.q_target.load_state_dict(self.q.state_dict())


def test():
    env = MakeEnv()
    agent = AgentDQN(4, 4)
    # env = gym.make('CartPole-v1')
    # agent = AgentDQN(4, 2)
    print_n = 20
    all_scores = []

    print(f"Running test")
    for n_epi in range(10000):
        s, done, score = env.reset(), False, 0.0
        while not done:
            a = agent.get_action(s)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.add_mem_step((s,a,r,s_prime, done_mask))
            s = s_prime
            score += r

        all_scores.append(score)
        if agent.memory.size() > 1000:
            agent.train()
        if n_epi % print_n == 1:
            env.render()
            lib.plot(all_scores)
            print(f"n: {n_epi} -> Score: {score} -> eps: {agent.eps} -> Buffer_n: {agent.memory.size()}")
            agent.update_parameters()


if __name__ == "__main__":
    test()




