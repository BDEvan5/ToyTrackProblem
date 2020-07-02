import gym
import collections
import random
import sys
import numpy as np 
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import LibFunctions as lib



#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
h_size = 512
BATCH_SIZE = 64


EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'
name20 = 'DataRecords/TrainTrack1020.npy'


class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class ValueNet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ValueNet, self).__init__()

        self.fc1 = nn.Linear(obs_space + action_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrainModDQN:
    def __init__(self, obs_space, action_space, name="best_avg"):
        self.model = Qnet(obs_space, action_space)
        self.target = Qnet(obs_space, action_space)
        self.value_model = ValueNet(obs_space, action_space)
        self.value_target = ValueNet(obs_space, action_space)
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

class TestModDQN:
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



def TrainAgent(track_name, agent_name, buffer, i=0, load=True):
    env = TestEnv(track_name)
    agent = TrainDQN(env.state_space, env.action_space, agent_name)
    agent.try_load(load)

    print_n = 20
    rewards = []
    for n in range(201):
        score, done, state = 0, False, env.reset()
        while not done:
            a = agent.learning_act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            buffer.put((state, a, r, s_prime, done_mask))
            state = s_prime
            score += r
            agent.experience_replay(buffer)
        rewards.append(score)

        if n % print_n == 1:
            env.render()    
            exp = agent.exploration_rate
            mean = np.mean(rewards[-20:])
            b = buffer.size()
            print(f"Run: {n} --> Score: {score} --> Mean: {mean} --> exp: {exp} --> Buf: {b}")

    agent.save()
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("DataRecords/Training_DQN" + str(i))

    return rewards

# run training
def RunDQNTraining():
    track_name = name50
    agent_name = "DQNtrain2"
    buffer = ReplayBuffer()
    total_rewards = []

    collect_observations(buffer, track_name, 5000)
    # TrainAgent(track_name, agent_name, buffer, 0, False)
    for i in range(1, 100):
        print(f"Running batch: {i}")
        rewards = TrainAgent(track_name, agent_name, buffer, i, True)
        total_rewards += rewards

if __name__ == "__main__":
    RunDQNTraining()
    

