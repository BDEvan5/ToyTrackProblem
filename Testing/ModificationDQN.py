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

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class ValueNet(nn.Module):
    def __init__(self, obs_space):
        super(ValueNet, self).__init__()
        h_vals = int(h_size / 2)
        self.fc1 = nn.Linear(obs_space + 1, h_vals)
        self.fc2 = nn.Linear(h_vals, h_vals)
        self.fc3 = nn.Linear(h_vals, 1)
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrainModDQN:
    def __init__(self, obs_space, action_space, name="best_avg"):
        self.model = None
        self.target = None
        self.value_model = None
        self.value_target = None

        self.action_space = action_space
        self.obs_space = obs_space
        self.mod_space = 2

        self.update_steps = 0
        self.name = name

    def learning_mod_act(self, obs):
        if random.random() < self.model.exploration_rate:
            return random.randint(0, self.mod_space-1)
        else: 
            return self.act(obs)

    def mod_act(self, obs):
        out = self.model.forward(obs)
        return out.argmax().item()

    def full_action(self, obs):
        pp_action = self.get_pursuit_action(obs)
        obs_t = torch.from_numpy(obs).float()
        value_obs = np.append(obs, pp_action)
        value_obs_t = torch.from_numpy(value_obs).float()
        safe_value = self.value_model.forward(value_obs_t)
        # 0= crash, 1 = no crash
        threshold = 0.5
        if safe_value.detach().item() > threshold:
            action = pp_action
            system = 0 
            mod_action = None
        else:
            mod_action = self.learning_mod_act(obs_t) # [0, 1]
            action_modifier = 2 if mod_action == 1 else -2
            action = pp_action + action_modifier # swerves left or right
            action = np.clip(action, 0, self.action_space-1)
            system = 1

        return action, system, mod_action

    def get_pursuit_action(self, obs):
        try:
            grad = obs[1] / obs[0] # y/x
        except:
            grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.action_space - 1)
        action = int(angle / dth)

        return action

    def get_pp_acts(self, obses):
        arr = np.zeros((len(obses)))
        for i, obs in enumerate(obses):
            arr[i] = self.get_pursuit_action(obs)

        return arr


    def experience_replay(self, buffer0, buffer1):
        n_train = 1
        for i in range(n_train):
            if buffer1.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer1.memory_sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

        for i in range(n_train):
            if buffer0.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer0.memory_sample(BATCH_SIZE)

            pp_act = self.get_pp_acts(s_p.numpy())
            cat_act = torch.from_numpy(pp_act[:, None]).float()
            obs = torch.cat([s_p, cat_act], dim=1)
            next_values = self.value_target.forward(obs)
            # max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * next_values * done
            obs = torch.cat([s_p, a.float()], dim=1)
            q_vals = self.value_model.forward(obs)
            # q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_vals, q_update.detach())

            self.value_model.optimizer.zero_grad()
            loss.backward()
            self.value_model.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
            self.value_target.load_state_dict(self.value_model.state_dict())
        if self.update_steps % 12 == 1:
            self.model.exploration_rate *= EXPLORATION_DECAY 
            self.model.exploration_rate = max(EXPLORATION_MIN, self.model.exploration_rate)
            self.value_model.exploration_rate *= EXPLORATION_DECAY 
            self.value_model.exploration_rate = max(EXPLORATION_MIN, self.value_model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        # torch.save(self.model.state_dict(), '%s/%s_model.pth' % (directory, filename))
        # torch.save(self.target.state_dict(), '%s/%s_target.pth' % (directory, filename))
        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))
        torch.save(self.value_model, '%s/%s_Vmodel.pth' % (directory, filename))
        torch.save(self.value_target, '%s/%s_Vtarget.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        # self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        # self.target.load_state_dict(torch.load('%s/%s_target.pth' % (directory, filename)))
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))
        self.value_model = torch.load('%s/%s_Vmodel.pth' % (directory, filename))
        self.value_target = torch.load('%s/%s_Vtarget.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                pass
        else:
            self.model = Qnet(self.obs_space, self.mod_space)
            self.target = Qnet(self.obs_space, self.mod_space)
            self.value_model = ValueNet(self.obs_space)
            self.value_target = ValueNet(self.obs_space)
            print(f"Not loading - restarting training")

class TestModDQN:
    def __init__(self, obs_space, act_space, name="best_avg"):
        self.model = Qnet(obs_space, act_space)
        self.name = name

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, filename)))
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
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
    

