import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib

from ModelsRL import TD3


class FullAgentBase:
    def __init__(self, name, load):
        self.env_map = None
        self.path_name = None
        self.wpts = None
        self.deltas = None
        self.pind = None

        self.steps = 0

        self.agent = TD3(14, 2, 1, name)
        self.agent.try_load(load)

        self.max_v = 7.5
        self.max_d = 0.4

        self.v_history = []
        self.d_history = []
        self.v_ref_history = []
        self.d_ref_history = []

    def transform_obs(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_d]
        vr_scale = [(v_ref-1)/self.max_v * 2 -1]
        dr_scale = [d_ref/self.max_d]

        new_obs = np.concatenate([cur_v, cur_d, vr_scale, dr_scale, obs[5:]])
        return new_obs

    def show_history(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.d_history)
        plt.plot(self.d_ref_history)
        plt.title("D history comparison")
        plt.legend(['NN', 'Ref'])
        plt.ylim([-0.5, 0.5])
        plt.pause(0.001)

        plt.figure(6)
        plt.clf()
        plt.plot(self.v_history)
        plt.plot(self.v_ref_history)
        plt.title("V history comparison")
        plt.legend(['NN', 'Ref'])
        plt.ylim([0, 10])
        plt.pause(0.001)

        plt.figure(3)
        plt.clf()
        plt.plot(self.reward_history, 'x', markersize=15)
        plt.plot(self.critic_history)
        plt.title("Reward history vs critic")
        plt.pause(0.001)

    def reset_lap(self):
        self.reward_history.clear()
        self.d_history.clear()
        self.v_history.clear()
        self.d_ref_history.clear()
        self.v_ref_history.clear()
        self.critic_history.clear()
        self.prev_s = 0
          
    def get_target_references(self, obs):
        target, self.pind = self.env_map.find_target(obs)
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
        v_ref = min(v_ref, self.max_v)

        v_ref = np.clip(v_ref, -8, 8)
        delta_ref = np.clip(delta_ref, -0.4, 0.4)

        return v_ref, delta_ref

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
        self.env_map.set_wpts(self.wpts)

        return self.wpts

    def trasform_action(self, nn_action):
        v_ret = (nn_action[0] +1)*self.max_v/2+1
        d_ret = nn_action[1]* self.max_d
        ret_action = [v_ret, d_ret]

        return ret_action


class RefGenVehicleTrain(FullAgentBase):
    def __init__(self, name, load):
        FullAgentBase.__init__(self, name, load)

        self.last_action = None
        self.last_obs = None
        self.max_act_scale = 0.4
        self.prev_s = 0

        self.reward_history = []
        self.critic_history = []
        
    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        v_ref, d_ref = self.get_target_references(obs)
        self.v_ref_history.append(v_ref)
        self.d_ref_history.append(d_ref)

        self.last_obs = obs

        nn_action = self.agent.act(nn_obs)
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.last_action = nn_action

        ret_action = self.trasform_action(nn_action)
        self.v_history.append(ret_action[0])
        self.d_history.append(ret_action[1])

        self.steps += 1

        return ret_action

    def add_memory_entry(self, buffer, reward, s_prime, done):
        # new_reward = self.update_reward_curvature(reward, self.last_action, s_prime)
        new_reward = self.update_reward_deviation(reward, self.last_action, s_prime)
        # new_reward = self.update_reward_progress(reward, self.last_action, s_prime)

        nn_obs = self.transform_obs(self.last_obs)
        s_p_nn = self.transform_obs(s_prime)

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
            v_ref, d_ref = self.get_target_references(obs)
            action = self.trasform_action(action)
            v_dif = abs(v_ref - action[0])
            d_dif = abs(d_ref - action[1])

            new_reward = 0.2 - d_dif*0.5 #- v_dif * 0.02

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
        self.critic_history.append(self.agent.get_critic_value(nn_obs, action))
        self.last_action = action

        ret_action = [(action[0] +1)*self.max_v/2+1, action[1]* self.max_d]

        self.steps += 1

        return ret_action


