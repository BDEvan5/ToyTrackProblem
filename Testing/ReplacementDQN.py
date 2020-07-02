import gym
import collections
import random
import sys
import numpy as np 
from matplotlib import pyplot as plt
import timeit
import guppy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TrainEnv import TrainEnv
import LibFunctions as lib

#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)
    
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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


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
      

class TrainDQN:
    def __init__(self, obs_space, action_space, name="best_avg"):
        self.model = Qnet(obs_space, action_space)
        self.target = Qnet(obs_space, action_space)
        self.action_space = action_space
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.update_steps = 0
        self.name = name

    def learning_act(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            return self.act(obs)

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def experience_replay(self, buffer):
        n_train = 1
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

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

    def save(self, directory="./dqn_saves"):
        filename = self.name
        torch.save(self.model.state_dict(), '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target.state_dict(), '%s/%s_target.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        self.target.load_state_dict(torch.load('%s/%s_target.pth' % (directory, filename)))
        print(f"Loaded Agent: {self.name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                pass
        else:
            print(f"Not loading - restarting training")

class TestDQN:
    def __init__(self, obs_space, act_space, name="best_avg"):
        self.model = Qnet(obs_space, act_space)
        self.name = name

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        print(f"Loaded Agent: {self.name}")

    def act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        a = out.argmax().item()
        return a



def DebugAgentTraining(agent, env):
    rewards = []
    
    # observe_myenv(env, agent, 5000)
    for n in range(5000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay()
            env.box_render()
            
        rewards.append(score)

        exp = agent.exploration_rate
        mean = np.mean(rewards[-20:])
        b = len(agent.buffer)
        print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
        env.render()
        agent.save()


def collect_observations(env, agent, n_itterations=10000):
    s, done = env.reset(), False
    for i in range(n_itterations):
        # action = env.random_action()
        action = env.action_space.sample()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        agent.buffer.append((s, action, r, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print(" ")



def TrainAgent(agent, env):
    print_n = 20
    rewards = []
    # collect_observations(env, agent, 5000)
    for n in range(2000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay()
        rewards.append(score)

        if ps.virtual_memory().free < 5e3:
            print(f"Memory Error: Breaking --> {ps.virtual_memory().free}")
            break

        if n % print_n == 1:
            # env.render()
            exp = agent.exploration_rate
            mean = np.mean(rewards[-20:])
            b = len(agent.buffer)
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            # agent.save()

            # lib.plot(rewards, figure_n=2)
            # plt.figure(2).savefig("Training_" + agent.name)

    agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("Training_DQN")

def noFrillsAgentTraining(agent, env):
    print_n = 20
    h = guppy.hpy()
    rewards = []
    collect_observations(env, agent, 5000)
    trainings_steps = 0
    for n in range(2000):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            agent.buffer.append((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            trainings_steps += 1
            agent.experience_replay()
        rewards.append(score)

        if n % print_n == 1:
            # env.render()
            exp = agent.exploration_rate
            mean = np.mean(rewards[-20:])
            b = len(agent.buffer)
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")
            print(f"Training steps: {trainings_steps}")
            trainings_steps = 0
            print(f"{h.heap()}")

    # agent.save()
    # lib.plot(rewards, figure_n=2)
    # plt.figure(2).savefig("Training_DQN")

# run training
def RunDQNTraining():
    # env = TrainEnv(name20)
    # agent = TrainDQN(env.state_space, env.action_space, "TestDQN1")

    env = gym.make('CartPole-v1')
    agent = TrainDQN(4, 2, "TestDQN-CartPole")

    TrainAgent(agent, env)


    # agent = TrainDQN(env.state_space, env.action_space, "DebugDQN")

    # DebugAgentTraining(agent, env)

def experience_replay(n_itterations=1000):
    env = gym.make('CartPole-v1')
    agent = TrainDQN(4, 2, "TestDQN-CartPole")
    collect_observations(env, agent, 1000)
    for i in range(n_itterations):
        agent.experience_replay()
        print("\rExperiencing Replay {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()
    print("")


def run_CartPoleTest():
    env = gym.make('CartPole-v1')
    agent = TrainDQN(4, 2, "TestDQN-CartPole")
    # collect_observations(env, agent, 50000)

    # for i in range(100):
    #     print(timeit.timeit(experience_replay, number=1))

    noFrillsAgentTraining(agent, env)
    TrainAgent(agent, env)
  

if __name__ == "__main__":
    run_CartPoleTest()
    # RunDQNTraining()
    
