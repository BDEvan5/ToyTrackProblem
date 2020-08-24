import numpy as np 
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PathFinder import PathFinder, modify_path
import LibFunctions as lib
from CommonTestUtils import ReplayBufferDQN

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
      

class Vnet(nn.Module):
    def __init__(self, obs_space):
        super(Vnet, self).__init__()
        
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, 1)

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
            obs_t = torch.from_numpy(obs).float()
            a = self.greedy_action(obs_t)
            return [a]

    def greedy_action(self, obs):
        out = self.model.forward(obs)
        self.last_out = out
        return out.argmax().item()

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
        if self.update_steps % 12 == 1:
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


class TrainWillemDecideDQN:
    def __init__(self, obs_space, name):
        self.obs_space = obs_space
        self.name = name

        self.model = None
        self.target = None

        self.update_steps = 0

    def decide(self, obs):
        v = self.model.forward(obs)
        return v

    def train_episodes(self, buffer):
        n_train = 1
        for i in range(n_train):
            if buffer.size() < BATCH_SIZE:
                return
            s, v, r, s_p, done = buffer.memory_sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)

            g = torch.ones_like(done) * GAMMA
            q_update = r + g * next_values * done
            q_vals = self.model.forward(s)

            loss = F.mse_loss(q_vals, q_update.detach())

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

        self.update_networks()

        return loss.detach().numpy()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
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
        self.model = Vnet(self.obs_space)
        self.target = Vnet(self.obs_space)
        print(f"Created new model")


class WillemsVehicle:
    def __init__(self, env_map, name, obs_space, action_space, load=True):
        self.env_map = env_map
        self.wpts = None

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
        self.pind = 1
        self.target = None

        self.obs_space = obs_space
        self.action_space = action_space
        self.center_act = (self.action_space - 1) / 2
        self.agent = TrainWillemModDQN(obs_space, action_space, name)

        self.check_agent = TrainWillemDecideDQN(11, 'Decisions')
        self.check_agent.try_load(True)

        self.agent.try_load(load)
        self.state_action = None
        self.state_value = None

    def init_plan(self):
        fcn = self.env_map.obs_free_hm._check_line
        path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
        path = path_finder.run_search(10)
        # self.env_map.obs_hm.show_map(path)
        path = modify_path(path)
        self.wpts = path
        np.save(self.path_name, self.wpts)
        print("Path Generated")

        self.wpts = np.append(self.wpts, self.env_map.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self.env_map.race_course._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        # self.env_map.race_course.show_map(self.wpts)

        self.pind = 1

        return self.wpts

    def init_straight_plan(self):
        # this is when there are no known obs for training.
        start = self.env_map.start
        end = self.env_map.end

        resolution = 10
        dx, dy = lib.sub_locations(end, start)

        n_pts = max((round(max((abs(dx), abs(dy))) / resolution), 3))
        ddx = dx / (n_pts - 1)
        ddy = dy / (n_pts - 1)

        self.wpts = []
        for i in range(n_pts):
            wpt = lib.add_locations(start, [ddx, ddy], i)
            if not self.env_map.race_course._check_location(wpt):
                self.wpts.append(wpt)
            else:
                pass

        self.pind = 1

        return self.wpts

    def act(self, obs):
        v_ref, phi_ref = self.get_target_references(obs)

        """This is where the agent can be removed if needed"""
        nn_obs = self.transform_obs(obs, v_ref, phi_ref)
        nn_action = self.agent.act(nn_obs)
        v_ref, phi_ref_nn = self.modify_references(nn_action, v_ref, phi_ref)
        self.state_action = [nn_obs, nn_action]

        a, d_dot = self.control_system(obs, v_ref, phi_ref_nn)

        return [a, d_dot]

    def no_mod_act(self, obs):
        v_ref, phi_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs, phi_ref=phi_ref)
        nn_obs_t = torch.from_numpy(nn_obs).float()
        nn_value = self.check_agent.model(nn_obs_t)
        self.state_value = [nn_obs, nn_value]

        a, d_dot = self.control_system(obs, v_ref, phi_ref)

        return [a, d_dot]


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

    def add_memory_entry(self, reward, done, s_prime, buffer):
        new_reward = self.update_reward(reward, self.state_action[1])

        v_ref, d_ref = self.get_target_references(s_prime)
        nn_s_prime = self.transform_obs(s_prime, v_ref, d_ref)
        done_mask = 0.0 if done else 1.0

        mem_entry = (self.state_action[0], self.state_action[1], new_reward, nn_s_prime, done_mask)

        if new_reward == -1 or np.random.random() < 0.2: # save 20% of 1s
        # if new_reward != 1:
            buffer.put(mem_entry)

    def add_decision_entry(self, reward, done, s_prime, buffer):
        v_ref, phi_ref = self.get_target_references(s_prime)
        nn_s_prime = self.transform_obs(s_prime, phi_ref=phi_ref)
        done_mask = 0.0 if done else 1.0

        mem_entry = (self.state_value[0], self.state_value[1], reward, nn_s_prime, done_mask)

        buffer.put(mem_entry)


    def update_reward(self, reward, action):
        beta = 0.01
        d_action = abs(action[0] - self.center_act)
        if reward == -1:
            new_reward = -1
        elif d_action == 0:
            new_reward = 0
        else:
            new_reward = 0 - d_action * beta

        return new_reward

    def transform_obs(self, obs, v_ref=None, phi_ref=None):
        max_angle = np.pi/2
        max_v = 7.5

        # target_theta = (lib.get_bearing(obs[0:2], self.target) - obs[2]) / (2*max_angle)
        # nn_obs = [target_theta, obs[3]/max_v, obs[4]/max_angle, v_ref/max_v, d_ref/max_angle]
        # nn_obs = np.array(nn_obs)

        scaled_target_phi = phi_ref / max_angle
        nn_obs = [scaled_target_phi]

        nn_obs = np.concatenate([nn_obs, obs[5:]])

        return nn_obs

    def modify_references(self, nn_action, v_ref, phi_ref):
        d_phi = 0.4 # rad
        phi_new = phi_ref + (nn_action[0] - self.center_act) * d_phi

        v_ref_mod = v_ref

        return v_ref_mod, phi_new

    def get_target_references(self, obs):
        self._set_target(obs)

        v_ref = 7.5

        th_target = lib.get_bearing(obs[0:2], self.target)
        phi_ref = th_target - obs[2]
        phi_ref = lib.limit_theta(phi_ref)

        return v_ref, phi_ref

    def control_system(self, obs, v_ref, phi_ref):
        kp_a = 10
        a = (v_ref - obs[3]) * kp_a

        theta_dot = phi_ref * 1
        L = 0.33
        d_ref = np.arctan(theta_dot * L / max(((obs[3], 1))))
        
        kp_delta = 1
        d_dot = (d_ref - obs[4]) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 10
        if dis_cur_target < shift_distance and self.pind < len(self.wpts)-1: # how close to say you were there
            self.pind += 1
        
        self.target = self.wpts[self.pind]






