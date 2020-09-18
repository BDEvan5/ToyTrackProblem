import numpy as np 
import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TrajectoryPlanner import MinCurvatureTrajectory
import LibFunctions as lib

#Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
h_size = 512
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99


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
      

class TrainWillemModDQN:
    def __init__(self, obs_space, action_space, name):
        self.action_space = action_space
        self.obs_space = obs_space
        self.name = name

        self.model = None
        self.target = None
        self.last_out = None

        self.update_steps = 0

    def act(self, obs):
        if random.random() < self.model.exploration_rate:
            return [random.randint(0, self.action_space-1)]
        else: 
            a = self.greedy_action(obs)
            return a

    def get_out(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)

        return out.detach().numpy()

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        self.last_out = out
        a = [out.argmax().item()]
        return a

    def train_modification(self, buffer):
        """This is for single examples with no future values"""
        n_train = 5
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, a, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            target = r.detach().float()
            a = torch.squeeze(a, dim=-1)

            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)

            loss = F.mse_loss(q_a, target)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
        self.update_networks()

    def train_episodes(self, buffer):
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
            a = torch.squeeze(a, dim=-1)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 30 == 1:
            self.model.exploration_rate *= EXPLORATION_DECAY 
            self.model.exploration_rate = max(EXPLORATION_MIN, self.model.exploration_rate)

    def save(self, directory="./dqn_saves"):
        filename = self.name
        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))
        print(f"Saved Agent: {self.name}")

    def load(self, directory="./dqn_saves"):
        filename = self.name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))
        print(f"Loaded Agent: {self.name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                self.create_model()
        else:
            self.create_model()

    def create_model(self):
        self.model = Qnet(self.obs_space, self.action_space)
        self.target = Qnet(self.obs_space, self.action_space)
        print(f"Created new model")


class ModBaseVehicle:
    def __init__(self, name, obs_space, action_space, load):
        self.env_map = None
        self.wpts = None
        self.path_name = None

        self.pind = 1
        self.target = None
        self.steps = 0
        self.slow_freq = 1

        self.obs_space = obs_space
        self.action_space = action_space
        self.center_act = int((self.action_space - 1) / 2)
        self.state_action = None
        self.cur_nn_act = None
        self.prev_nn_act = self.center_act
        self.mem_window = [0, 0, 0, 0, 0]

        self.agent = TrainWillemModDQN(obs_space+5, action_space, name)
        self.agent.try_load(load)

        self.mod_history = []
        self.out_his = []
        self.reward_history = []

    def act(self, obs, greedy=False):
        v_ref, phi_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs, v_ref, phi_ref)
        """This is where the agent can be removed if needed"""
        agent_on = True
        if agent_on:

            if not greedy:
                nn_action = self.agent.act(nn_obs)
            else:
                nn_action = self.agent.greedy_action(nn_obs)
            # nn_action = [1]
            self.cur_nn_act = nn_action
        else:
            self.cur_nn_act = [self.center_act]

        self.out_his.append(self.agent.get_out(nn_obs))
        self.mod_history.append(self.cur_nn_act)
        self.state_action = [nn_obs, self.cur_nn_act]

        self.mem_window.pop(0)
        self.mem_window.append(float(self.cur_nn_act[0]/3))

        v_ref, phi_ref = self.modify_references(self.cur_nn_act, v_ref, phi_ref)

        self.steps += 1

        a, d_dot = self.control_system(obs, v_ref, phi_ref)

        return [a, d_dot]

    def show_vehicle_history(self):
        lib.plot_no_avg(self.mod_history, figure_n=1, title="Mod history")
        # lib.plot_no_avg(self.reward_history, figure_n=2, title="Reward history")
        lib.plot_multi(self.out_his, "Outputs", figure_n=3)

        plt.figure(3)
        plt.plot(self.reward_history)

        self.mod_history.clear()
        self.out_his.clear()
        self.reward_history.clear()
        self.steps = 0

    def transform_obs(self, obs, v_ref=None, phi_ref=None):
        max_angle = np.pi/2
        max_v = 7.5

        scaled_target_phi = phi_ref / max_angle
        nn_obs = [scaled_target_phi]

        nn_obs = np.concatenate([nn_obs, obs[5:], self.mem_window])

        return nn_obs

    def modify_references(self, nn_action, v_ref, phi_ref):
        d_phi = 0.5 # rad
        phi_new = phi_ref + (nn_action[0] - self.center_act) * d_phi

        v_ref_mod = v_ref

        return v_ref_mod, phi_new

    def get_target_references(self, obs):
        self._set_target(obs)

        v_ref = 7.5

        th_target = lib.get_bearing(obs[0:2], self.target)
        phi_ref = lib.sub_angles_complex(th_target, obs[2])

        return v_ref, phi_ref

    def control_system(self, obs, v_ref, phi_ref):
        kp_a = 10
        a = (v_ref - obs[3]) * kp_a

        theta_dot = phi_ref * 1
        L = 0.33
        d_ref = np.arctan(theta_dot * L / max(((obs[3], 1))))
        
        kp_delta = 5
        d_dot = (d_ref - obs[4]) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance: # how close to say you were there
            self.pind += 1
            if self.pind == len(self.wpts)-1:
                self.pind = 1
        
        self.target = self.wpts[self.pind]

    def update_reward(self, reward, action):
        beta = 0.1
        d_action = abs(action[0] - self.center_act)
        if reward == -1:
            new_reward = -1
        # elif reward == 1:
        #     new_reward = 1
        elif d_action == 0:
            new_reward = 0
        else:
            dd_action = abs(action[0] - self.prev_nn_act)
            new_reward = 0 - d_action * beta - dd_action *beta

        self.reward_history.append(new_reward)

        return new_reward

    def add_memory_entry(self, reward, done, s_prime, buffer):
        if reward !=0 or self.steps % self.slow_freq == 0:
            new_reward = self.update_reward(reward, self.state_action[1])
            self.prev_nn_act = self.state_action[1][0]

            v_ref, d_ref = self.get_target_references(s_prime)
            nn_s_prime = self.transform_obs(s_prime, v_ref, d_ref)
            done_mask = 0.0 if done else 1.0

            mem_entry = (self.state_action[0], self.state_action[1], new_reward, nn_s_prime, done_mask)

            if new_reward != 0 or np.random.random() < 0.2: # save 20% of 1s
            # if new_reward != 1:
                buffer.put(mem_entry)


class ModTrainVehicle(ModBaseVehicle):
    def __init__(self, name, obs_space, action_space, load):
        ModBaseVehicle.__init__(self, name, obs_space, action_space, load)

    def init_plan(self, env_map):
        self.env_map = env_map
        track = self.env_map.track
        n_set = MinCurvatureTrajectory(track, self.env_map.obs_map)

        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        self.wpts = r_line

        self.pind = 1

        return self.wpts

    def random_act(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs, v_ref, d_ref)
        nn_action = [np.random.randint(0, self.action_space-1)]
        v_ref, d_ref = self.modify_references(nn_action, v_ref, d_ref)

        a, d_dot = self.control_system(obs, v_ref, d_ref)

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        self.state_action = [nn_obs, nn_action]

        return [a, d_dot]

    def reset_lap(self):
        self.pind = 1


class ModRaceVehicle(ModBaseVehicle):
    def __init__(self, name, obs_space, action_space, load):
        ModBaseVehicle.__init__(self, name, obs_space, action_space, load)

    def init_plan(self, env_map):
        self.env_map = env_map
        track = self.env_map.track
        n_set = MinCurvatureTrajectory(track, self.env_map.obs_map)

        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        self.wpts = r_line

        self.pind = 1

        return self.wpts

    def reset_lap(self):
        self.pind = 1



