import numpy as np 
import random
from matplotlib import pyplot as plt

from ModelsRL import TD3

import LibFunctions as lib
from Simulator import ScanSimulator



class BaseGenAgent:
    def __init__(self, name, n_beams):
        self.name = name
        self.env_map = None
        self.wpts = None

        self.path_name = None
        self.pind = 1
        self.target = None

        # history
        self.mod_history = []
        self.d_ref_history = []
        self.reward_history = []
        self.critic_history = []
        self.steps = 0

        self.max_v = 7.5
        self.max_d = 0.4
        self.max_friction_force = 3.74 * 9.81 * 0.523 *0.5

        # agent stuff 
        self.state_action = None
        self.cur_nn_act = None
        self.prev_nn_act = 0

        self.scan_sim = ScanSimulator(n_beams, np.pi)
        self.n_beams = n_beams

    def init_agent(self, env_map):
        self.env_map = env_map
        
        self.scan_sim.set_check_fcn(self.env_map.check_scan_location)

        # self.wpts = self.env_map.get_min_curve_path()
        self.wpts = self.env_map.get_reference_path()

        self.prev_dist_target = lib.get_distance(self.env_map.start, self.env_map.end)

        return self.wpts

    def show_vehicle_history(self):
        plt.figure(1)
        plt.clf()
        plt.title("Mod History")
        plt.ylim([-1.1, 1.1])
        plt.plot(self.mod_history)
        np.save('Vehicles/mod_hist', self.mod_history)
        # plt.plot(self.d_ref_history)
        plt.legend(['NN'])

        plt.pause(0.001)

        # plt.figure(3)
        # plt.clf()
        # plt.title('Rewards')
        # plt.ylim([-1.5, 4])
        # plt.plot(self.reward_history, 'x', markersize=12)
        # plt.plot(self.critic_history)

    def transform_obs(self, obs):
        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_d]

        th_target = lib.get_bearing(obs[0:2], self.env_map.end)
        th_scale = [(th_target)*2/np.pi]

        scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])

        nn_obs = np.concatenate([cur_v, cur_d, th_scale, scan])

        return nn_obs

    def generate_references(self, nn_action, obs):
        d = nn_action[0] * self.max_d
        d_ref = np.clip(d, - self.max_d, self.max_d)

        d_plan = max(abs(d_ref), abs(obs[4]))
        theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
        v_ref = self.max_friction_force / (3.74 * max(theta_dot, 0.01)) 
        v_ref = min(v_ref, 8.5)

        return v_ref, d_ref

    def reset_lap(self):
        self.mod_history.clear()
        self.d_ref_history.clear()
        self.reward_history.clear()
        self.critic_history.clear()
        self.steps = 0
        self.pind = 1


class GenVehicleTrainDistance(BaseGenAgent):
    def __init__(self, name, load, h_size, n_beams):
        BaseGenAgent.__init__(self, name, n_beams)

        self.prev_dist_target = 0

        state_space = 3 + self.n_beams
        self.agent = TD3(state_space, 1, 1, name)
        self.agent.try_load(load, h_size)

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs)
        self.cur_nn_act = nn_action

        self.mod_history.append(self.cur_nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.cur_nn_act]

        v_ref, d_ref = self.generate_references(self.cur_nn_act, obs)

        self.steps += 1

        return [v_ref, d_ref]

    def update_reward(self, reward, s_prime):
        beta = 1
        if reward == -1:
            new_reward = -1
            self.prev_dist_target = lib.get_distance(self.env_map.start, self.env_map.end)
        else:
            dist_target = lib.get_distance(s_prime[0:2], self.env_map.end)
            new_reward = (self.prev_dist_target - dist_target) * beta
            self.prev_dist_target = dist_target

        self.reward_history.append(new_reward)

        return new_reward

    def add_memory_entry(self, reward, done, s_prime, buffer):
        new_reward = self.update_reward(reward, s_prime)
        self.prev_nn_act = self.state_action[1][0]

        nn_s_prime = self.transform_obs(s_prime)
        # done_mask = 0.0 if done else 1.0

        mem_entry = (self.state_action[0], self.state_action[1], nn_s_prime, new_reward, done)

        buffer.add(mem_entry)

        return new_reward


class GenVehicleTrainSteering(BaseGenAgent):
    def __init__(self, name, load, h_size, n_beams):
        BaseGenAgent.__init__(self, name, n_beams)

        self.prev_dist_target = 0

        state_space = 3 + self.n_beams
        self.agent = TD3(state_space, 1, 1, name)
        self.agent.try_load(load, h_size)

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs)
        self.cur_nn_act = nn_action

        self.mod_history.append(self.cur_nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.cur_nn_act]

        v_ref, d_ref = self.generate_references(self.cur_nn_act, obs)

        self.steps += 1

        return [v_ref, d_ref]

    def update_reward(self, reward, s_prime):
        beta = 2
        if reward == -1:
            new_reward = -1
            self.prev_dist_target = lib.get_distance(self.env_map.start, self.env_map.end)
        else:
            new_reward = 1 - abs(s_prime[4]) * beta

        self.reward_history.append(new_reward)

        return new_reward

    def add_memory_entry(self, reward, done, s_prime, buffer):
        new_reward = self.update_reward(reward, s_prime)
        self.prev_nn_act = self.state_action[1][0]

        nn_s_prime = self.transform_obs(s_prime)
        # done_mask = 0.0 if done else 1.0

        mem_entry = (self.state_action[0], self.state_action[1], nn_s_prime, new_reward, done)

        buffer.add(mem_entry)

        return new_reward


class GenVehicleTest(BaseGenAgent):
    def __init__(self, name):
        path = 'Vehicles/' + name + ''
        state_space = 4 
        self.agent = TD3(state_space, 1, 1, name)
        self.agent.load(directory=path)

        print(f"NN: {self.agent.actor.type}")

        nn_size = self.agent.actor.l1.in_features
        n_beams = nn_size - 4
        BaseGenAgent.__init__(self, name, n_beams)

        self.current_v_ref = None
        self.current_phi_ref = None

    def act(self, obs):
        v_ref, d_ref = self.get_target_references(obs)

        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs, noise=0)
        self.cur_nn_act = nn_action

        self.d_ref_history.append(d_ref)
        self.mod_history.append(self.cur_nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.cur_nn_act]

        v_ref, d_ref = self.modify_references(self.cur_nn_act, v_ref, d_ref, obs)

        self.steps += 1

        return [v_ref, d_ref]
