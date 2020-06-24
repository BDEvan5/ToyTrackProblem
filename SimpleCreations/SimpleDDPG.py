import numpy as np
from collections import deque
import random
import gym

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from SimpleEnv import MakeEnv
import sys
import LibFunctions as lib
from matplotlib import  pyplot as plt

tau = 0.01
gamma = 0.99

class BasicBuffer_a:
    def __init__(self, size, obs_dim, act_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = np.asarray([rew])
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict= dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])
        return (temp_dict['s'],temp_dict['a'],temp_dict['r'].reshape(-1,1),temp_dict['s2'],temp_dict['d'])

class Critic_gen(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Critic_gen, self).__init__()

        self.fc1 = nn.Linear(state_dim + act_dim, 1204)
        self.fc2 = nn.Linear(1204, 512)
        self.fc3 = nn.Linear(512, 300)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        obs = torch.cat(x, axis=-1)

        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        v =  self.fc4(h3)

        return v

class Actor_gen(nn.Module):
    def __init__(self, state_dim, act_dim, max_act=1):
        super(Actor_gen, self).__init__()

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 128)
        self.fc4 = nn.Linear(128, act_dim)

        self.max_act = max_act

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        a =  torch.tanh(self.fc4(h3))

        act = torch.mul(a, self.max_act)

        return act


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_act):
        self.obs_dim = state_dim
        self.action_dim = action_dim # x and y for target
        self.action_max = max_act # will be scaled later
        
        # Main network outputs
        self.mu = Actor_gen(state_dim, action_dim, max_act)
        self.q_mu = Critic_gen(state_dim, action_dim)

        # Target networks
        self.mu_target = Actor_gen(state_dim, action_dim, max_act)
        self.q_mu_target = Critic_gen(state_dim, action_dim)
      
        # Copying weights in,
        self.mu.load_state_dict(self.mu_target.state_dict())
        self.q_mu_target.load_state_dict(self.q_mu.state_dict())
    
        # optimizers
        self.mu_optimizer = torch.optim.Adam(self.mu.parameters(), lr=1e-3)
        self.q_mu_optimizer = torch.optim.Adam(self.q_mu.parameters(), lr=1e-3)

        self.replay_buffer = BasicBuffer_a(100000, obs_dim=state_dim, act_dim=action_dim)
        
        self.q_losses = []
        self.mu_losses = []
        
    def act(self, s, noise_scale):
        s = torch.tensor(s, dtype=torch.float)
        a =  self.mu(s).detach().numpy()
        a += noise_scale * np.random.randn(self.action_dim)
        act = np.clip(a, -self.action_max, self.action_max)

        return act

    def train(self):
        batch_size = 32
        if self.replay_buffer.size < batch_size:
            return 
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)

        X = torch.tensor(X, dtype=torch.float)
        A = torch.tensor(A, dtype=torch.float)
        X2 = torch.tensor(X2, dtype=torch.float)
        R = torch.tensor(R, dtype=torch.float)

        # Updating Ze Critic
        A2 =  self.mu_target(X2)
        q_target = R + gamma * self.q_mu_target([X2,A2]).detach()
        qvals = self.q_mu([X,A]) 
        q_loss = ((qvals - q_target)**2).mean()
        self.q_mu_optimizer.zero_grad()
        q_loss.backward()
        self.q_mu_optimizer.step()
        self.q_losses.append(q_loss)

        #Updating ZE Actor
        A_mu =  self.mu(X)
        Q_mu = self.q_mu([X,A_mu])
        mu_loss =  - Q_mu.mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        soft_update(self.mu, self.mu_target)
        soft_update(self.q_mu, self.q_mu_target)

    def save(self, filename="best_avg", directory="./saves"):
        torch.save(self.mu.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.q_mu.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.mu.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.q_mu.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def test_gym():
    env = gym.make("Pendulum-v0")
    agent = DDPGAgent(env.observation_space.shape[0], 1, 2)

    episode_rewards = []

    noise = 0.1
    for episode in range(20):
        state = env.reset()
        episode_reward = 0

        for step in range(500):
            action = agent.act(state, noise)

            next_state, reward, done, _ = env.step(action)
            d_store = False if step == 499 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            agent.train()   

            if done or step == 499:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


def observe(env,replay_buffer, observation_steps):
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        # action = env.action_space.sample()
        action = env.random_action()
        new_obs, reward, done, _ = env.step_continuous(action)

        replay_buffer.push(obs, action, reward, new_obs, done)

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


def RunMyEnv(agent_name, show=True):
    env = MakeEnv()
    # env.add_obstacles(3)
    agent = DDPGAgent(env.state_dim, env.action_dim, env.max_action)

    show_n = 2

    rewards = []
    observe(env, agent.replay_buffer, 40000)
    for i in range(500):
        agent.train()
    agent.save(agent_name)
    return

    for episode in range(80):
        score, done, obs, ep_steps = 0, False, env.reset(), 0
        while not done:
            action = agent.act(np.array(obs), 0.1)

            new_obs, reward, done, _ = env.step_continuous(action) 
            done_bool = 0 if ep_steps + 1 == 200 else float(done)
        
            agent.replay_buffer.push(obs, action, reward, new_obs, done_bool)          
            obs = new_obs
            score += reward
            ep_steps += 1

            agent.train() # number is of itterations

        rewards.append(score)
        if show:
            print(f"Ep: {episode} -> score: {score}")
            if episode % show_n == 1:
                lib.plot(rewards, figure_n=2)
                plt.figure(2).savefig("Training_" + agent_name)
                env.render()
                # agent.save(agent_name)

    agent.save(agent_name)
    lib.plot(rewards, figure_n=2)
    plt.figure(2).savefig("Training_" + agent_name)

def eval2(agent_name, show=True):
    env = MakeEnv()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.max_action)

    show_n = 1

    rewards = []
    try:
        agent.load(agent_name)
        print(f"Agent loaded: {agent_name}")
    except:
        print("Cannot load agent")

    for episode in range(100):
        score, done, obs, ep_steps = 0, False, env.reset(), 0
        while not done:
            action = agent.act(np.array(obs), 0)

            new_obs, reward, done, _ = env.step_continuous(action) 
            done_bool = 0 if ep_steps + 1 == 200 else float(done)
        
            agent.replay_buffer.push(obs, action, reward, new_obs, done_bool)       
            obs = new_obs
            score += reward
            ep_steps += 1

        rewards.append(score)
        if show:
            print(f"Ep: {episode} -> score: {score}")
            if episode % show_n == 1:
                lib.plot(rewards)
                env.render()

    print(f"Avg reward: {np.mean(rewards)}")



if __name__ == "__main__":
    # test_gym()
    agent_name = "TestingDDPG"
    RunMyEnv(agent_name)
    eval2(agent_name, False)

