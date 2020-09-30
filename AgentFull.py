import numpy as np 
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import LibFunctions as lib


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2



class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, name):
        self.name = name
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.act_dim = action_dim

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, iterations):
        for it in range(iterations):
            # Sample replay buffer 
            x, u, y, r, d = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, POLICY_NOISE)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * GAMMA * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % POLICY_FREQUENCY == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory="./saves"):
        filename = self.name
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

    def load(self, directory="./saves"):
        filename = self.name
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename)))

        print("Agent Loaded")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                pass
        else:
            # self.create_agent()
            print(f"Not loading - restarting training")



class FullAgentBase:
    def __init__(self, name, load):
        self.env_map = None
        self.path_name = None
        self.wpts = None

        self.steps = 0

        self.agent = TD3(12, 2, 1, name)
        self.agent.try_load(load)

         
    def transform_obs(self, obs):
        # _, d_ref = self.get_target_references(obs)
        # new_obs = np.append(obs[5:], d_ref)
        # new_obs = np.concatenate([new_obs, self.mem_window])

        # new_obs = obs[5:]

        new_obs = np.concatenate([[obs[3]/7.5], [obs[4]/0.4], obs[5:]])
        return new_obs


class FullTrainVehicle(FullAgentBase):
    def __init__(self, name, load):
        FullAgentBase.__init__(self, name, load)

        self.last_action = None
        self.last_obs = None
        self.max_act_scale = 0.4
        self.prev_s = 0

        self.reward_history = []
        self.v_history = []
        self.d_history = []

        self.mem_window = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        

    def init_agent(self, env_map):
        self.env_map = env_map


    def act_cs(self, obs):
        nn_obs = self.transform_obs(obs)

        self.last_obs = obs

        action = self.agent.select_action(nn_obs)
        # self.mem_window.pop(0)
        # self.mem_window.append(d_ref_nn)
        self.v_history.append(action[0])
        self.d_history.append(action[1])
        self.last_action = action

        max_v = 6.5
        max_d = 0.4
        ret_action = [(action[0] +1)*max_v/2+1, action[1]* max_d]

        self.steps += 1

        return ret_action

    def add_memory_entry(self, buffer, reward, s_prime, done):
        new_reward = self.update_reward(reward, self.last_action, s_prime)

        s_p_nn = self.transform_obs(s_prime)
        nn_obs = self.transform_obs(self.last_obs)

        mem_entry = (nn_obs, self.last_action, s_p_nn, new_reward, done)

        buffer.add(mem_entry)

        return new_reward

    def update_reward(self, reward, action, obs):
        if reward == -1:
            new_reward = -1
        else:
            # v_beta = 0.02
            # d_beta = 0.1
            
            # new_reward = 0.05 - abs(action[1]) * d_beta  + v_beta * (action[0] + 1)/2 
            s_beta = 0.8
            s = self.env_map.get_s_progress(obs[0:2])
            ds = s - self.prev_s
            ds = np.clip(ds, -0.5, 0.5)
            new_reward = ds * s_beta
            self.prev_s = s


        self.reward_history.append(new_reward)

        return new_reward



    def show_history(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.d_history)
        plt.plot(self.v_history)
        plt.title("NN v & d hisotry")
        plt.legend(['d his', 'v_his'])
        plt.ylim([-1.1, 1.1])
        plt.pause(0.001)

        plt.figure(3)
        plt.clf()
        plt.plot(self.reward_history, 'x', markersize=15)
        plt.title("Reward history")
        plt.pause(0.001)

    def reset_lap(self):
        self.reward_history.clear()
        self.d_history.clear()
        self.v_history.clear()
        self.prev_s = 0

