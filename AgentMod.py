import numpy as np 
import random
from matplotlib import pyplot as plt

from ModelsRL import DQN

import LibFunctions as lib




class BaseModAgent:
    def __init__(self, name, load):
        self.name = name
        self.env_map = None
        self.wpts = None

        self.path_name = None
        self.pind = 1
        self.target = None

        # history
        self.mod_history = []
        self.out_his = []
        self.reward_history = []
        self.steps = 0

        # agent stuff 
        self.action_space = 5
        self.center_act = int((self.action_space - 1) / 2)
        self.state_action = None
        self.cur_nn_act = None
        self.prev_nn_act = self.center_act
        self.mem_window = [0, 0, 0, 0, 0]

        self.agent = DQN(11, self.action_space, name)
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

    def show_vehicle_history(self):
        lib.plot_no_avg(self.mod_history, figure_n=1, title="Mod history")
        # lib.plot_no_avg(self.reward_history, figure_n=2, title="Reward history")
        lib.plot_multi(self.out_his, "Outputs", figure_n=3)

        plt.figure(3)
        plt.plot(self.reward_history, linewidth=2)

    def transform_obs(self, obs, v_ref=None, phi_ref=None):
        max_angle = np.pi/2
        max_v = 7.5

        scaled_target_phi = phi_ref / max_angle
        nn_obs = [scaled_target_phi]

        # nn_obs = np.concatenate([nn_obs, obs[5:], self.mem_window])
        nn_obs = np.concatenate([nn_obs, obs[5:]])

        return nn_obs

    def modify_references(self, nn_action, v_ref, d_ref, obs):
        d_phi = 0.5 / self.center_act# rad
        d_new = d_ref + (nn_action[0] - self.center_act) * d_phi

        if abs(d_new) > abs(d_ref):
            max_friction_force = 3.74 * 9.81 * 0.523 *0.5
            d_plan = max(abs(d_ref), abs(obs[4]), abs(d_new))
            theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
            v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
            v_ref_mod = min(v_ref, 8.5)
        else:
            v_ref_mod = v_ref


        return v_ref_mod, d_new

    def reset_lap(self):
        self.mod_history.clear()
        self.out_his.clear()
        self.reward_history.clear()
        self.steps = 0
        self.pind = 1


class ModVehicleTrain(BaseModAgent):
    def __init__(self, name, load):
        BaseModAgent.__init__(self, name, load)

        self.current_v_ref = None
        self.current_phi_ref = None

        self.mem_save = True

    def act(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs, v_ref, d_ref)
        nn_action = self.agent.act(nn_obs)
        self.cur_nn_act = nn_action

        self.out_his.append(self.agent.get_out(nn_obs))
        self.mod_history.append(self.cur_nn_act)
        self.state_action = [nn_obs, self.cur_nn_act]

        # self.mem_window.pop(0)
        # self.mem_window.append(float(self.cur_nn_act[0]/self.action_space)) # normalises it.

        v_ref, d_ref = self.modify_references(self.cur_nn_act, v_ref, d_ref, obs)

        self.steps += 1

        return [v_ref, d_ref]

    def update_reward(self, reward, action):
        beta = 0.1 / self.center_act
        d_action = abs(action[0] - self.center_act)
        if reward == -1:
            new_reward = -1
        else:
            dd_action = abs(action[0] - self.prev_nn_act)
            new_reward = 0.05 - d_action * beta - dd_action *beta/2
            # new_reward = 0.01

        self.reward_history.append(new_reward)

        return new_reward

    def add_memory_entry(self, reward, done, s_prime, buffer):
        # if self.mem_save:
        if True:
            new_reward = self.update_reward(reward, self.state_action[1])
            self.prev_nn_act = self.state_action[1][0]

            v_ref, d_ref = self.get_target_references(s_prime)
            nn_s_prime = self.transform_obs(s_prime, v_ref, d_ref)
            done_mask = 0.0 if done else 1.0

            mem_entry = (self.state_action[0], self.state_action[1], new_reward, nn_s_prime, done_mask)

            # if new_reward != 0 or np.random.random() < 0.2: 
            buffer.put(mem_entry)

            # self.mem_save = False
        return new_reward


class ModVehicleTest(BaseModAgent):
    def __init__(self, name, load):
        BaseModAgent.__init__(self, name, load)

        self.current_v_ref = None
        self.current_phi_ref = None

        self.mem_save = True

    def act(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs, v_ref, d_ref)
        nn_action = self.agent.greedy_action(nn_obs)
        self.cur_nn_act = nn_action

        self.out_his.append(self.agent.get_out(nn_obs))
        self.mod_history.append(self.cur_nn_act)
        self.state_action = [nn_obs, self.cur_nn_act]

        # self.mem_window.pop(0)
        # self.mem_window.append(float(self.cur_nn_act[0]/self.action_space)) # normalises it.

        v_ref, d_ref = self.modify_references(self.cur_nn_act, v_ref, d_ref, obs)

        self.steps += 1

        return [v_ref, d_ref]
