import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PathFinder import PathFinder, modify_path
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



class PureRepDataGen:
    def __init__(self, env_map):
        self.env_map = env_map
        
        self.wpts = None
        self.pind = 1
        self.target = None

        self.nn_wpts = None
        self.nn_pind = 1
        self.nn_target = None

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call

    def init_agent(self):
        fcn = self.env_map.obs_hm._check_line
        path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
        path = None
        while path is None:
            try:
                path = path_finder.run_search(5)
            except AssertionError:
                print(f"Search Problem: generating new start")
                self.env_map.generate_random_start()

        self.wpts = modify_path(path)
        # print("Path Generated")

        self.wpts = np.append(self.wpts, self.env_map.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self.env_map.race_course._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        self.env_map.race_course.show_map(False, self.wpts)

        self.pind = 1

        self.init_straight_plan()        

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

        self.nn_wpts = []
        for i in range(n_pts):
            pt = lib.add_locations(start, [ddx, ddy], i)
            self.nn_wpts.append(pt)

        self.nn_pind = 1

    def act(self, obs):
        # v_ref, d_ref = self.get_corridor_references(obs)
        self._set_target(obs)
        v_ref, d_ref = self.get_target_references(obs, self.target)
        a, d_dot = self.control_system(obs, v_ref, d_ref)

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return [a, d_dot]

    def get_nn_vals(self, obs):
        v_ref, d_ref = self.get_target_references(obs, self.nn_target)

        max_angle = np.pi/2
        max_v = 7.5

        target_theta = (lib.get_bearing(obs[0:2], self.nn_target) - obs[2]) / (2*max_angle)
        nn_obs = [target_theta, obs[3]/max_v, obs[4]/max_angle, v_ref/max_v, d_ref/max_angle]
        nn_obs = np.array(nn_obs)

        nn_obs = np.concatenate([nn_obs, obs[5:]])

        return nn_obs

    def get_corridor_references(self, obs):
        ranges = obs[5:]
        max_range = np.argmax(ranges)
        dth = np.pi / 9
        theta_dot = dth * max_range - np.pi/2

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        v_ref = 6

        return v_ref, delta_ref

    def get_target_references(self, obs, target):
        v_ref = 6

        th_target = lib.get_bearing(obs[0:2], target)
        theta_dot = th_target - obs[2]
        theta_dot = lib.limit_theta(theta_dot)

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        return v_ref, delta_ref

    def control_system(self, obs, v_ref, d_ref):
        kp_a = 10
        a = (v_ref - obs[3]) * kp_a
        
        kp_delta = 5
        d_dot = (d_ref - obs[4]) * kp_delta

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance and self.pind < len(self.wpts)-2: # how close to say you were there
            self.pind += 1
        
        self.target = self.wpts[self.pind]

        dis_cur_target = lib.get_distance(self.nn_wpts[self.nn_pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance and self.nn_pind < len(self.nn_wpts)-2: # how close to say you were there
            self.nn_pind += 1
        
        self.nn_target = self.nn_wpts[self.nn_pind]


class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        y = F.relu(self.l2(x))
        z = self.l3(y)
        a = self.max_action * torch.tanh(z) 
        return a



class SuperTrainRep(object):
    def __init__(self, state_dim, action_dim, agent_name):
        self.model = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_name = agent_name

    def train(self, replay_buffer):
        s, a_r = replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(s)
        right_actions = torch.FloatTensor(a_r)

        actions = self.model(states)

        actor_loss = F.mse_loss(actions, right_actions)

        self.model.optimizer.zero_grad()
        actor_loss.backward()
        self.model.optimizer.step()

        return actor_loss

    def save(self, directory='./td3_saves'):
        torch.save(self.model, '%s/%s_model.pth' % (directory, self.agent_name))

    def load(self, directory='./td3_saves'):
        self.model = torch.load('%s/%s_model.pth' % (directory, self.agent_name))

    def create_agent(self):
        self.model = Actor(self.state_dim, self.action_dim, 1)

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                pass
        else:
            self.create_agent()
            print(f"Not loading - restarting training")

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.model(state).data.numpy().flatten()

        return action


