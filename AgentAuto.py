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

    def load(self, directory="./saves"):
        filename = self.name
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

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



class AutoAgentBase:
    def __init__(self, name, load):
        self.env_map = None
        self.path_name = None
        self.wpts = None

        self.pind = 1
        self.target = None
        self.steps = 0

        self.current_v_ref = None
        self.current_d_ref = None

        self.agent = TD3(10, 1, 0.4, name)
        self.agent.try_load(load)

    def init_agent(self, env_map):
        self.env_map = env_map
        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
 
        self.wpts = self.env_map.get_min_curve_path()

        r_line = self.wpts
        ths = [lib.get_bearing(r_line[i], r_line[i+1]) for i in range(len(r_line)-1)]
        alphas = [lib.sub_angles_complex(ths[i+1], ths[i]) for i in range(len(ths)-1)]
        lds = [lib.get_distance(r_line[i], r_line[i+1]) for i in range(1, len(r_line)-1)]

        self.deltas = np.arctan(2*0.33*np.sin(alphas)/lds)

        self.pind = 1

        return self.wpts
         
    def get_target_references(self, obs):
        # self._set_target(obs)
        self.find_target(obs)

        target = self.wpts[self.pind]
        th_target = lib.get_bearing(obs[0:2], target)
        alpha = lib.sub_angles_complex(th_target, obs[2])

        # pure pursuit
        ld = lib.get_distance(obs[0:2], target)
        delta_ref = np.arctan(2*0.33*np.sin(alpha)/ld)

        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)

        max_friction_force = 3.74 * 9.81 * 0.523 *0.5
        d_plan = max(abs(delta_ref), abs(obs[4]), max_d)
        theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
        v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
        v_ref = min(v_ref, 8.5)
        v_ref = 2

        v_ref = np.clip(v_ref, -8, 8)
        delta_ref = np.clip(delta_ref, -0.4, 0.4)

        return v_ref, delta_ref

    def control_system(self, obs):
        v_ref = self.current_v_ref
        d_ref = self.current_d_ref

        kp_a = 10
        a = (v_ref - obs[3]) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - obs[4]) * kp_delta

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 1
        while dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
                dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
            else:
                self.pind = 0

    def find_target(self, obs):
        distances = [lib.get_distance(obs[0:2], self.wpts[i]) for i in range(len(self.wpts))]
        ind = np.argmin(distances)
        N = len(self.wpts)
        if ind == N:
            ind = 1

        front_dis = lib.get_distance(self.wpts[ind], self.wpts[ind+1])
        back_dis = lib.get_distance(self.wpts[ind], self.wpts[ind-1])

        if front_dis < back_dis * 0.5:
            self.pind = ind + 1
        else:
            self.pind = ind
        # self.pind = ind + 1


    def transform_obs(self, obs):
        # _, d_ref = self.get_target_references(obs)
        # new_obs = np.append(obs[5:], d_ref)
        # new_obs = np.concatenate([new_obs, self.mem_window])

        new_obs = obs[5:]
        return new_obs


class AutoTrainVehicle(AutoAgentBase):
    def __init__(self, name, load):
        AutoAgentBase.__init__(self, name, load)

        self.last_action = None
        self.last_obs = None
        self.max_act_scale = 0.4

        self.reward_history = []
        self.nn_history = []
        self.d_ref_hisotry = []

        self.mem_window = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        v_ref, d_ref = self.get_target_references(obs)
        self.current_v_ref = v_ref
        self.d_ref_hisotry.append(d_ref)
        self.last_obs = nn_obs

        self.current_d_ref = self.agent.select_action(nn_obs)[0]
        self.mem_window.pop(0)
        self.mem_window.append(self.current_d_ref)
        self.nn_history.append(self.current_d_ref)
        self.last_action = self.current_d_ref

        a, d_dot = self.control_system(obs)

        self.steps += 1

        a = np.clip(a, -8.5, 8.5)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return [a, d_dot]

    def act_cs(self, obs):
        nn_obs = self.transform_obs(obs)
        v_ref, d_ref = self.get_target_references(obs)
        self.current_v_ref = v_ref
        self.d_ref_hisotry.append(d_ref)
        self.last_obs = obs

        d_ref_nn = self.agent.select_action(nn_obs)[0]
        # self.mem_window.pop(0)
        # self.mem_window.append(d_ref_nn)
        self.nn_history.append(d_ref_nn)
        self.last_action = d_ref_nn

        self.steps += 1

        return [v_ref, d_ref_nn]

    def add_memory_entry(self, buffer, reward, s_prime, done):
        # new_reward = self.update_reward(reward, self.last_action)
        new_reward = self.update_reward_dev(reward, self.last_action, self.last_obs)

        s_p_nn = self.transform_obs(s_prime)
        nn_obs = self.transform_obs(self.last_obs)

        mem_entry = (nn_obs, [self.last_action], s_p_nn, new_reward, done)

        buffer.add(mem_entry)

        return new_reward

    def update_reward(self, reward, action):
        if reward == -1:
            new_reward = -1
        else:
            new_reward = 0.01 - abs(action) * 0.005

        self.reward_history.append(new_reward)

        return new_reward

    def update_reward_dev(self, reward, action, obs):
        if reward == -1:
            new_reward = -1
        else:
            v_, d_ref = self.get_target_references(obs)
            d_dif = abs(d_ref - action)
            new_reward = 0.5 - d_dif *2 

        self.reward_history.append(new_reward)

        return new_reward

    def show_history(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.d_ref_hisotry)
        plt.plot(self.nn_history)
        plt.title("NN and d ref hisotry")
        plt.legend(['D_ref', 'NN'])
        plt.ylim([-0.5, 0.5])
        plt.pause(0.001)

        plt.figure(3)
        plt.clf()
        plt.plot(self.reward_history)
        plt.title("Reward history")
        plt.pause(0.001)

    def reset_lap(self):
        self.pind = 1
        self.reward_history.clear()
        self.nn_history.clear()
        self.d_ref_hisotry.clear()

