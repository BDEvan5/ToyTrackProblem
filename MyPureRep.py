import numpy as np 
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from PathFinder import PathFinder, modify_path
from MinCurveTrajPlanner import MinCurvatureTrajectory
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
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 300)
        self.l3 = nn.Linear(300, 100)
        self.l4 = nn.Linear(100, action_dim)

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        y = F.relu(self.l2(x))
        z = F.relu(self.l3(y))
        w = self.l4(z)
        a = self.max_action * torch.tanh(w) 

        return a

"""The agent class which is trained"""
class SuperTrainRep(object):
    def __init__(self, state_dim, action_dim, agent_name):
        self.model = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_name = agent_name

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print(f"Device: {self.device}")

    def train(self, replay_buffer, iters=5):
        if len(replay_buffer.storage) > BATCH_SIZE:
            for i in range(iters):
                s, a_r = replay_buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(s).to(self.device)
                right_actions = torch.FloatTensor(a_r).to(self.device)

                actions = self.model(states)

                actor_loss = F.mse_loss(actions, right_actions)

                self.model.optimizer.zero_grad()
                actor_loss.backward()
                self.model.optimizer.step()

            return actor_loss.detach().item()
        return 0

    def save(self, directory='./td3_saves'):
        torch.save(self.model, '%s/%s_model.pth' % (directory, self.agent_name))
        print(f"Agent saved: {self.agent_name}")

    def load(self, directory='./td3_saves'):
        self.model = torch.load('%s/%s_model.pth' % (directory, self.agent_name))
        print(f"The agent has loaded: {self.agent_name}")

    def create_agent(self):
        self.model = Actor(self.state_dim, self.action_dim, 1)
        print(f"Agent created: {self.state_dim}, {self.action_dim}")

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
        self.model.to(self.device)

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.model(state).data.numpy().flatten()

        return action


class RepBaseVehicle:
    def __init__(self, agent_name, load):
        self.wpts = None
        self.pind = 1
        self.target = None

        self.agent = SuperTrainRep(11 + 5, 1, agent_name)
        self.agent.try_load(load)

        self.mem_window = [0, 0, 0, 0, 0]

        self.nn_phi_history = []
        self.train_phi_history = []

        self.env_map = None
        self.path_name = None

    def act(self, obs):
        self._set_targets(obs)
        
        v_ref = 6
        nn_obs = self.get_nn_vals(obs)
        nn_act = self.agent.act(nn_obs)[0] 
        self.nn_phi_history.append(nn_act)

        # # add target to record: for display only
        # v_ref, target_phi = self.get_target_references(obs, self.train_target)
        # self.target_phi_history.append(target_phi/ np.pi *2)

        self.mem_window.pop(0)
        self.mem_window.append(float(nn_act))

        nn_phi = nn_act * np.pi/2

        a, d_dot = self.control_system(obs, v_ref, nn_phi)

        return [a, d_dot]

    def show_history(self):
        plt.figure(1)
        plt.clf()        
        plt.title('History')
        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(self.nn_phi_history)
        plt.plot(self.train_phi_history)

        plt.legend(['NN', 'Target'])
        plt.ylim([-1.1, 1.1])

        plt.pause(0.001)

    def get_nn_vals(self, obs):
        v_ref, target_phi_straight = self.get_target_references(obs, self.target)

        max_angle = np.pi

        scaled_target_phi = target_phi_straight / max_angle
        nn_obs = [scaled_target_phi]

        nn_obs = np.concatenate([nn_obs, obs[5:], self.mem_window])

        return nn_obs

    def get_target_references(self, obs, target):
        v_ref = 6

        th_target = lib.get_bearing(obs[0:2], target)
        target_phi = th_target - obs[2]
        target_phi = lib.limit_theta(target_phi)

        return v_ref, target_phi

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





class RepTrainVehicle(RepBaseVehicle):
    def __init__(self, agent_name, load):
        RepBaseVehicle.__init__(self, agent_name, load)

        self.train_wpts = None
        self.train_pind = 1
        self.train_target = None
      
    # def init_plan(self, env_map=None):
    #     if env_map is not None:
    #         self.env_map = env_map
    #         self.path_name = "Maps/" + self.env_map.name + "_path.npy"

    #     # self.env_map.obs_hm.show_map(True)
    #     fcn = self.env_map.obs_hm._check_line
    #     path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
    #     path = None
    #     while path is None:
    #         try:
    #             path = path_finder.run_search(5)
    #         except AssertionError:
    #             # self.env_map.obs_hm.show_map(True)
    #             print(f"Search Problem: generating new start")
    #             self.env_map.reset_map()

    #     self.wpts = modify_path(path)

    #     self.wpts = np.append(self.wpts, self.env_map.end)
    #     self.wpts = np.reshape(self.wpts, (-1, 2))

    #     new_pts = []
    #     for wpt in self.wpts:
    #         if not self.env_map.race_course._check_location(wpt):
    #             new_pts.append(wpt)
    #         else:
    #             pass
    #     self.wpts = np.asarray(new_pts)    

    #     # self.env_map.race_course.show_map(False, self.wpts)

    #     self.init_train_plan()     
    #     self.reset_lap()   

    #     return self.wpts

    # def init_train_plan(self):
    #     fcn = self.env_map.obs_hm._check_line
    #     path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
    #     path = None
    #     while path is None:
    #         try:
    #             path = path_finder.run_search(5)
    #         except AssertionError:
    #             print(f"Search Problem: generating new start")
    #             self.env_map.generate_random_start()

    #     self.train_wpts = modify_path(path)

    #     self.train_wpts = np.append(self.train_wpts, self.env_map.end)
    #     self.train_wpts = np.reshape(self.train_wpts, (-1, 2))

    #     new_pts = []
    #     for wpt in self.train_wpts:
    #         if not self.env_map.race_course._check_location(wpt):
    #             new_pts.append(wpt)
    #         else:
    #             pass
    #     self.train_wpts = np.asarray(new_pts)    

    #     # self.env_map.race_course.show_map(False, self.train_wpts)     

    #     return self.wpts

    def init_plan(self, env_map):
        self.env_map = env_map
        track = env_map.track
        n_set = MinCurvatureTrajectory(track, env_map.obs_map)

        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        self.wpts = r_line

        self.train_wpts = r_line # using the same wpts

        self.train_phi_history.clear()
        self.nn_phi_history.clear()

        self.pind = 1
        self.train_pind = 1

        return self.wpts

    def reset_lap(self):
        self.pind = 1
        self.train_pind = 1

        self.train_phi_history.clear()
        self.nn_phi_history.clear()

    def train_act(self, obs):
        self._set_targets(obs)

        v_ref, target_phi = self.get_target_references(obs, self.train_target)
        normalised_target_phi = target_phi/ np.pi *2
        self.train_phi_history.append(normalised_target_phi)

        # record values
        nn_obs = self.get_nn_vals(obs)
        nn_act = self.agent.act(nn_obs)[0] 
        self.nn_phi_history.append(nn_act)

        self.mem_window.pop(0)
        self.mem_window.append(float(nn_act))

        a, d_dot = self.control_system(obs, v_ref, target_phi)

        return [a, d_dot]

    def add_mem_step(self, buffer, obs):
        nn_state = self.get_nn_vals(obs)
        v, target_phi = self.get_target_references(obs, self.train_target)
        data = (nn_state, [target_phi/np.pi*2])
        buffer.add(data)

    def _set_targets(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance: # how close to say you were there
            self.pind += 1
            if self.pind == len(self.wpts)-1:
                self.pind = 1
        
        self.target = self.wpts[self.pind]

        dis_cur_target = lib.get_distance(self.train_wpts[self.train_pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance: # how close to say you were there
            self.train_pind += 1
            if self.train_pind == len(self.train_wpts)-1:
                self.train_pind = 1
        
        self.train_target = self.train_wpts[self.train_pind]


class RepRaceVehicle(RepBaseVehicle):
    def __init__(self, agent_name, load):
        RepBaseVehicle.__init__(self, agent_name, load)

    def init_plan(self, env_map=None):
        if env_map is not None:
            self.env_map = env_map
            self.path_name = "Maps/" + self.env_map.name + "_path.npy"

        fcn = self.env_map.obs_hm._check_line
        path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
        path = None
        while path is None:
            try:
                path = path_finder.run_search(5)
            except AssertionError:
                print(f"Search Problem: generating new start")
                self.env_map.reset_map()

        self.wpts = modify_path(path)

        self.wpts = np.append(self.wpts, self.env_map.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self.env_map.race_course._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        # self.env_map.race_course.show_map(False, self.wpts)    
        self.reset_lap()

        return self.wpts

    def _set_targets(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance: # how close to say you were there
            self.pind += 1
            if self.pind == len(self.wpts)-1:
                self.pind = 1
        
        self.target = self.wpts[self.pind]

    def reset_lap(self):
        self.pind = 1



# class RepTrainVehicle:
#     def __init__(self, env_map, agent_name="Testing", load=True):
#         self.env_map = env_map
        
#         self.wpts = None
#         self.pind = 1
#         self.target = None

#         self.nn_wpts = None
#         self.nn_pind = 1
#         self.nn_target = None

#         self.agent = SuperTrainRep(11+5, 1, agent_name)
#         self.agent.try_load(load)

#         self.mem_window = [0, 0, 0, 0, 0]

#         self.nn_phi_history = []
#         self.target_phi_history = []
        

#         self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call

#     # def init_agent(self):
#     #     fcn = self.env_map.obs_hm._check_line
#     #     path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
#     #     path = None
#     #     while path is None:
#     #         try:
#     #             path = path_finder.run_search(5)
#     #         except AssertionError:
#     #             print(f"Search Problem: generating new start")
#     #             self.env_map.generate_random_start()

#     #     self.wpts = modify_path(path)

#     #     self.wpts = np.append(self.wpts, self.env_map.end)
#     #     self.wpts = np.reshape(self.wpts, (-1, 2))

#     #     new_pts = []
#     #     for wpt in self.wpts:
#     #         if not self.env_map.race_course._check_location(wpt):
#     #             new_pts.append(wpt)
#     #         else:
#     #             pass
#     #     self.wpts = np.asarray(new_pts)    

#     #     # self.env_map.race_course.show_map(False, self.wpts)

#     #     self.pind = 1

#     #     self.target_phi_history.clear()
#     #     self.nn_phi_history.clear()

#     #     self.init_straight_plan()        

#     #     return self.wpts

#     def init_agent(self):
#         track = self.env_map.track
#         n_set = MinCurvatureTrajectory(track, self.env_map.obs_map)

#         deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
#         r_line = track[:, 0:2] + deviation
#         self.wpts = r_line

#         self.nn_wpts = r_line # using the same wpts

#         self.target_phi_history.clear()
#         self.nn_phi_history.clear()

#         self.pind = 1
#         self.nn_pind = 1

#         return self.wpts

#     # def init_straight_plan(self):
#     #     # this is when there are no known obs for training.
#     #     start = self.env_map.start
#     #     end = self.env_map.end

#     #     resolution = 10
#     #     dx, dy = lib.sub_locations(end, start)

#     #     n_pts = max((round(max((abs(dx), abs(dy))) / resolution), 3))
#     #     ddx = dx / (n_pts - 1)
#     #     ddy = dy / (n_pts - 1)

#     #     self.nn_wpts = []
#     #     for i in range(n_pts):
#     #         pt = lib.add_locations(start, [ddx, ddy], i)
#     #         self.nn_wpts.append(pt)

#     #     self.nn_pind = 1

#     #     self.target_phi_history.clear()
#     #     self.nn_phi_history.clear()

#     #     return self.nn_wpts

#     def opti_act(self, obs):
#         self._set_targets(obs)

#         v_ref, target_phi = self.get_target_references(obs, self.target)
#         normalised_target_phi = target_phi/ np.pi *2
#         self.target_phi_history.append(normalised_target_phi)

#         # record values
#         nn_obs = self.get_nn_vals(obs)
#         nn_act = self.agent.act(nn_obs)[0] 
#         self.nn_phi_history.append(nn_act)

#         self.mem_window.pop(0)
#         # self.mem_window.append(float(normalised_target_phi))
#         self.mem_window.append(float(nn_act))

#         a, d_dot = self.control_system(obs, v_ref, target_phi)

#         return [a, d_dot]

#     def act(self, obs):
#         self._set_targets(obs)
        
#         v_ref = 6
#         nn_obs = self.get_nn_vals(obs)
#         nn_act = self.agent.act(nn_obs)[0] 
#         self.nn_phi_history.append(nn_act)

#         # add target to record
#         v_ref, target_phi = self.get_target_references(obs, self.target)
#         self.target_phi_history.append(target_phi/ np.pi *2)

#         self.mem_window.pop(0)
#         self.mem_window.append(float(nn_act))

#         nn_phi = nn_act * np.pi/2

#         a, d_dot = self.control_system(obs, v_ref, nn_phi)


#         return [a, d_dot]

#     def show_history(self):
#         plt.figure(1)
#         plt.clf()        
#         plt.title('History')
#         plt.xlabel('Episode')
#         plt.ylabel('Duration')

#         plt.plot(self.nn_phi_history)
#         plt.plot(self.target_phi_history)

#         plt.legend(['NN', 'Target'])
#         plt.ylim([-1.1, 1.1])

#         plt.pause(0.001)

#     def get_nn_vals(self, obs):
#         v_ref, target_phi_straight = self.get_target_references(obs, self.env_map.end)

#         max_angle = np.pi

#         scaled_target_phi = target_phi_straight / max_angle
#         nn_obs = [scaled_target_phi]

#         nn_obs = np.concatenate([nn_obs, obs[5:], self.mem_window])

#         return nn_obs

#     def add_mem_step(self, buffer, obs):
#         nn_state = self.get_nn_vals(obs)
#         v, target_phi = self.get_target_references(obs, self.target)
#         data = (nn_state, [target_phi/np.pi*2])
#         buffer.add(data)

#     def get_target_references(self, obs, target):
#         v_ref = 6

#         th_target = lib.get_bearing(obs[0:2], target)
#         target_phi = th_target - obs[2]
#         target_phi = lib.limit_theta(target_phi)

#         return v_ref, target_phi

#     def control_system(self, obs, v_ref, phi_ref):
#         kp_a = 10
#         a = (v_ref - obs[3]) * kp_a

#         theta_dot = phi_ref * 1
#         L = 0.33
#         d_ref = np.arctan(theta_dot * L / max(((obs[3], 1))))
        
#         kp_delta = 5
#         d_dot = (d_ref - obs[4]) * kp_delta

#         a = np.clip(a, -8, 8)
#         d_dot = np.clip(d_dot, -3.2, 3.2)

#         return a, d_dot

#     def _set_targets(self, obs):
#         dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
#         shift_distance = 5
#         if dis_cur_target < shift_distance and self.pind < len(self.wpts)-2: # how close to say you were there
#             self.pind += 1
        
#         self.target = self.wpts[self.pind]

#         dis_cur_target = lib.get_distance(self.nn_wpts[self.nn_pind], obs[0:2])
#         shift_distance = 5
#         if dis_cur_target < shift_distance and self.nn_pind < len(self.nn_wpts)-2: # how close to say you were there
#             self.nn_pind += 1
        
#         self.nn_target = self.nn_wpts[self.nn_pind]


# class RepRaceVehicle:
#     def __init__(self, env_map, agent_name="Testing", load=True):
#         self.env_map = env_map
        
#         self.wpts = None
#         self.pind = 1
#         self.target = None

#         self.agent = SuperTrainRep(11 + 5, 1, agent_name)
#         self.agent.try_load(load)

#         self.mem_window = [0, 0, 0, 0, 0]

#         self.nn_phi_history = []
#         self.target_phi_history = []
        
#         self.path_name = 'Maps/' + self.env_map.name + '_path.npy' # move to setup call

#     def init_race_plan(self):
#         self.env_map.obs_free_hm.show_map(False)
#         fcn = self.env_map.obs_free_hm._check_line
#         path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
#         path = None

#         try:
#             path = np.load(self.path_name)
#         except:
#             path = path_finder.run_search(5)
#             np.save(self.path_name, path)

#         self.wpts = modify_path(path)

#         self.wpts = np.append(self.wpts, self.env_map.end)
#         self.wpts = np.reshape(self.wpts, (-1, 2))

#         new_pts = []
#         for wpt in self.wpts:
#             if not self.env_map.race_course._check_location(wpt):
#                 new_pts.append(wpt)
#             else:
#                 pass
#         self.wpts = np.asarray(new_pts)    

#         # self.env_map.race_course.show_map(False, self.wpts)

#         self.pind = 1

#         self.nn_phi_history.clear() 

#         return self.wpts

#     def reset_lap_count(self):
#         self.pind = 1

#     def act(self, obs):
#         self._set_targets(obs)
        
#         v_ref = 6
#         nn_obs = self.get_nn_vals(obs)
#         nn_act = self.agent.act(nn_obs)[0] 
#         self.nn_phi_history.append(nn_act)

#         self.mem_window.pop(0)
#         self.mem_window.append(float(nn_act))

#         nn_phi = nn_act * np.pi/2

#         a, d_dot = self.control_system(obs, v_ref, nn_phi)

#         return [a, d_dot]

#     def show_history(self):
#         plt.figure(1)
#         plt.clf()        
#         plt.title('History')
#         plt.xlabel('Episode')
#         plt.ylabel('Duration')

#         plt.plot(self.nn_phi_history)

#         plt.legend(['NN', 'Target'])
#         plt.ylim([-1.1, 1.1])

#         plt.pause(0.001)

#     def get_nn_vals(self, obs):
#         v_ref, target_phi_straight = self.get_target_references(obs, self.target)

#         max_angle = np.pi

#         scaled_target_phi = target_phi_straight / max_angle
#         nn_obs = [scaled_target_phi]

#         nn_obs = np.concatenate([nn_obs, obs[5:], self.mem_window])

#         return nn_obs

#     def get_target_references(self, obs, target):
#         v_ref = 6

#         th_target = lib.get_bearing(obs[0:2], target)
#         target_phi = th_target - obs[2]
#         target_phi = lib.limit_theta(target_phi)

#         return v_ref, target_phi

#     def control_system(self, obs, v_ref, phi_ref):
#         kp_a = 10
#         a = (v_ref - obs[3]) * kp_a

#         theta_dot = phi_ref * 1
#         L = 0.33
#         d_ref = np.arctan(theta_dot * L / max(((obs[3], 1))))
        
#         kp_delta = 5
#         d_dot = (d_ref - obs[4]) * kp_delta

#         a = np.clip(a, -8, 8)
#         d_dot = np.clip(d_dot, -3.2, 3.2)

#         return a, d_dot

#     def _set_targets(self, obs):
#         dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
#         shift_distance = 5
#         if dis_cur_target < shift_distance and self.pind < len(self.wpts)-2: # how close to say you were there
#             self.pind += 1
        
#         self.target = self.wpts[self.pind]



