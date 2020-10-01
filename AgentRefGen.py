import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib

from ModelsRL import TD3


class FullAgentBase:
    def __init__(self, name, load):
        self.env_map = None
        self.path_name = None
        self.wpts = None

        self.steps = 0

        self.agent = TD3(14, 2, 1, name)
        self.agent.try_load(load)

        self.max_v = 7.5
        self.max_d = 0.4

        self.env_map = None
        self.path_name = None 
        self.wpts = None
        self.deltas = None
        self.pind = None
       
    def transform_obs(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        v = self.max_v
        d = self.max_d

        new_obs = np.concatenate([[obs[3]/v], [obs[4]/d], [v_ref/v], [d_ref/d], obs[5:]])
        return new_obs

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
          
    def get_target_references(self, obs):
        self._set_target(obs)

        target = self.wpts[self.pind]
        th_target = lib.get_bearing(obs[0:2], target)
        alpha = lib.sub_angles_complex(th_target, obs[2])

        # pure pursuit
        ld = lib.get_distance(obs[0:2], target)
        delta_ref = np.arctan(2*0.33*np.sin(alpha)/ld)

        # ds = self.deltas[self.pind:self.pind+1]
        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)
        # max_d = max(abs(ds))

        max_friction_force = 3.74 * 9.81 * 0.523 *0.5
        d_plan = max(abs(delta_ref), abs(obs[4]), max_d)
        theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
        v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
        v_ref = min(v_ref, 8.5)
        # v_ref = 3

        return v_ref, delta_ref

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 1
        while dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
                dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
            else:
                self.pind = 0
          
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


class RefGenVehicleTrain(FullAgentBase):
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
        
    def act(self, obs):
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
        new_reward = self.update_reward_curvature(reward, self.last_action, s_prime)

        s_p_nn = self.transform_obs(s_prime)
        nn_obs = self.transform_obs(self.last_obs)

        mem_entry = (nn_obs, self.last_action, s_p_nn, new_reward, done)

        buffer.add(mem_entry)

        return new_reward

    def update_reward_steering(self, reward, action):
        if reward == -1:
            new_reward = -1
        else:
            new_reward = 0.1 - abs(action) * 0.2

        self.reward_history.append(new_reward)

        return new_reward

    def update_reward_deviation(self, reward, action, obs):
        if reward == -1:
            new_reward = -1
        else:
            v_, d_ref = self.get_target_references(obs)
            d_dif = abs(d_ref - action)
            new_reward = 0.5 - d_dif 

        self.reward_history.append(new_reward)

        return new_reward

    def update_reward_progress(self, reward, action, obs):
        if reward == -1:
            new_reward = -1
        else:
            s_beta = 0.8
            s = self.env_map.get_s_progress(obs[0:2])
            ds = s - self.prev_s
            ds = np.clip(ds, -0.5, 0.5)
            new_reward = ds * s_beta
            self.prev_s = s


        self.reward_history.append(new_reward)

        return new_reward      
          
    def update_reward_curvature(self, reward, action, obs):
        if reward == -1:
            new_reward = -1
        else:
            v_beta = 0.02
            d_beta = 0.1
            
            new_reward = 0.05 - abs(action[1]) * d_beta  + v_beta * (action[0] + 1)/2 

        self.reward_history.append(new_reward)

        return new_reward


class RefGenVehicleTest(FullAgentBase):
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

    def act(self, obs):
        nn_obs = self.transform_obs(obs)

        self.last_obs = obs

        action = self.agent.select_action(nn_obs, 0)

        self.v_history.append(action[0])
        self.d_history.append(action[1])
        self.last_action = action


        ret_action = [(action[0] +1)*self.max_v/2+1, action[1]* self.max_d]

        self.steps += 1

        return ret_action
