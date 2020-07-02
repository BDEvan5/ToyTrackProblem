import gym
import collections
import random
import sys
import numpy as np 
from matplotlib import pyplot as plt
import timeit
import memory_profiler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from EnvBase import TrainEnv
import LibFunctions as lib

#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995



class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        h_size = 512
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class DQN:
    def __init__(self, obs_space, action_space):
        self.model = Qnet(obs_space, action_space)
        self.target = Qnet(obs_space, action_space)
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.update_steps = 0

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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def sample_action(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def experience_replay(self):
        n_train = 1
        for i in range(n_train):
            if len(self.buffer) < BATCH_SIZE:
                return
            s, a, r, s_p, done = self.memory_sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self, filename="best_avg", directory="./dqn_saves"):
        torch.save(self.model.state_dict(), '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target.state_dict(), '%s/%s_target.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./dqn_saves"):
        self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        self.target.load_state_dict(torch.load('%s/%s_target.pth' % (directory, filename)))


def observe_myenv(env, agent, n_itterations=10000):
    s = env.reset()
    done = False
    for i in range(n_itterations):
        action = env.random_discrete_action()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        agent.buffer.append((s, action, r/100, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")

def RunMyEnv(agent_name, show=True):
    env = TrainEnv()
    # env.add_obstacles(6)
    agent = DQN(env.state_dim, env.action_space)

    print_n = 20
    show_n = 10

    rewards = []
    observe_myenv(env, agent, 5000)
    for n in range(5000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.sample_action(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r/100, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay()
            # env.box_render()
            
        rewards.append(score)

        if show:
            if n % print_n == 1:
                exp = agent.exploration_rate
                mean = np.mean(rewards[-20:])
                b = len(agent.buffer)
                print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
                # lib.plot(rewards, figure_n=2)
                # plt.figure(2).savefig("Training_" + agent_name)
            if n % show_n == 1:
                env.render()
                # env.render_actions()
                # agent.save(agent_name)

    agent.save(agent_name)
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("Training_" + agent_name)



if __name__ == '__main__':
    # test_cartpole()
    agent_name = "TestingDQN10"
    RunMyEnv(agent_name, True)

    # t = timeit.timeit(stmt=timing, number=1)
    # print(f"Time: {t}")
