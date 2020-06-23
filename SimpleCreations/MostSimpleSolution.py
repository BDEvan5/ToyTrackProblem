import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import collections
import random
import gym
import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib 

class MakeEnv:
    def __init__(self):
        self.car_x = [0, 0]
        self.last_distance = None
        self.start = None
        self.end = None

        self.x_bound = [1, 99]
        self.y_bound = [1, 99]
        self.steps = 0

    def reset(self):
        self.steps = 0
        rands = np.random.rand(4) * 100
        self.start = rands[0:2]
        self.end = rands[2:4]  

        while self._check_bounds(self.start):
            self.start = np.random.rand(2) * 100
        while self._check_bounds(self.end) or \
            lib.get_distance(self.start, self.end) < 15:
            self.end = np.random.rand(2) * 100

        self.last_distance = lib.get_distance(self.start, self.end)
        self.car_x = self.start

        return self._get_state_obs()

    def step(self, action):
        self.steps += 1
        new_x = self._new_x(action)
        if self._check_bounds(new_x):
            obs = self._get_state_obs()
            r_crash = -100
            return obs, r_crash, True, None
        self.car_x = new_x 
        obs = self._get_state_obs()
        reward, done = self._get_reward()
        return obs, reward, done, None


    def _get_state_obs(self):
        scale = 100
        rel_target = lib.sub_locations(self.end, self.car_x)
        obs = np.array(rel_target) / scale
        
        # current obs is in terms of an x, y target
        return obs

    def _get_reward(self):
        beta = 2
        r_done = 100

        cur_distance = lib.get_distance(self.car_x, self.end)
        if cur_distance < 5:
            return r_done, True
        reward = beta * (self.last_distance - cur_distance)
        self.last_distance = cur_distance
        done = self._check_steps()
        return reward, done

    def _check_steps(self):
        if self.steps > 100:
            return True
        return False

    def _check_bounds(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 
        return False

    def _new_x(self, action):
        # action is 0, 1, 2, 3
        if action == 0:
            dx = [0, 1]
        elif action == 1:
            dx = [0, -1]
        elif action == 2:
            dx = [1, 0]
        elif action == 3:
            dx = [-1, 0]
        x = lib.add_locations(dx, self.car_x)
        
        return x


#Hyperparameters
learning_rate = 0.005
gamma         = 0.98
buffer_limit  = 50000
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
    agent = AgentDQN(2, 4)
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
            agent.add_mem_step((s,a,r/100.0,s_prime, done_mask))
            s = s_prime
            score += r

        print(f"n: {n_epi} -> Score: {score} -> eps: {agent.eps} -> Buffer_n: {agent.memory.size()}")
        all_scores.append(score)
        lib.plot(all_scores)
        if agent.memory.size() > 1000:
            agent.train()
        if n_epi % print_n == 1:
            agent.update_parameters()


if __name__ == "__main__":
    test()




