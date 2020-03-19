import LocationState as ls
import numpy as np
import LibFunctions as f
import EpisodeMem as em
import TrackEnv1
import random


class Controller:
    def __init__(self, env, logger):
        self.env = env # at the moment the points are stored in the track - consider chanings
        self.logger = logger

        self.dt = 0.5 # controller frequency

        self.state = ls.CarState()
        self.control_sys = ControlSystem()
        self.agent_q = AgentQ(3, 4)

    def run_control(self):
        self.state = self.env.reset()
        for w_pt in self.env.track.route: # steps through way points
            i = 0
            ret = self.control_to_wp(w_pt)

            if not ret:
                print("Episode finished due to crash")
                break # it crashed

    def run_standard_control(self):
        # this is with no rl
        self.state = self.env.reset()
        for w_pt in self.env.track.route: # steps through way points
            i = 0
            ret = self.standard_control(w_pt)

            if not ret:
                break # it crashed

    def standard_control(self, wp):
        # no rl for testing
        i = 0
        max_steps = 100

        while not self._check_if_at_target(wp) and i < max_steps: #keeps going till at each pt
            
            action = self.control_sys.get_controlled_action(self.state, wp)
            next_state, reward, done = self.env.step(action)

            obs = self.state.get_sense_observation()
            next_obs = next_state.get_sense_observation()

            if done:
                return False
                # break
            i+=1 
        return True # got to wp

    def control_to_wp(self, wp):
        i = 0
        max_steps = 100

        while not self._check_if_at_target(wp) and i < max_steps: #keeps going till at each pt
            
            action = self.control_sys.get_controlled_action(self.state, wp)
            agent_action = self.check_action_rl(action)
            action, dr = self.get_new_action(agent_action, action)

            next_state, reward, done = self.env.step(action)

            reward += dr
            obs = self.state.get_sense_observation()
            next_obs = next_state.get_sense_observation()
            self.agent_q.update_q_table(obs, agent_action, reward, next_obs)

            self.state = next_state
            if done:
                return False
            i+=1 
        return True # got to wp

    def _check_if_at_target(self, wp):
        way_point_dis = f.get_distance(self.state.x, wp.x)
        # print(way_point_dis)
        if way_point_dis < 2:
            # print("Way point reached: " + str(wp.x))
            return True
        return False
    
    def check_action_rl(self, action):
        observation = self.state.get_sense_observation()
        agent_action = self.agent_q.get_action(observation)
        return agent_action

    def get_new_action(self, agent_action, action):
        theta_swerve = 0.8
        # interpret action
        dr = -20
        if agent_action == 1: # stay in the centre
            dr = 0
            return action, dr 
        if agent_action == 0: # swerve left
            action = [action[0], action[1] - theta_swerve]
            print("Swerving left")
            return action, dr
        if agent_action == 2: # swerve right
            action = [action[0], action[1] + theta_swerve]
            print("Swerving right")
            return action, dr


class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction

    def get_controlled_action(self, state, wp):
        x_ref = wp.x 
        v_ref = wp.v 
        th_ref = wp.theta

        # run v control
        e_v = v_ref - state.v # error for controler
        a = self._acc_control(e_v)

        # run th control
        x_ref_th = self._get_xref_th(state.x, x_ref)
        e_th = th_ref * self.k_th_ref + x_ref_th * (1- self.k_th_ref) # no feedback
        th = self._th_controll(e_th)

        action = [a, th]
        return action

    def _acc_control(self, e_v):
        # this function is the actual controller
        k = 0.15
        return k * e_v

    def _th_controll(self, e_th):
        # theta controller to come here when dth!= th
        return e_th

    def _get_xref_th(self, x1, x2):
        dx = x2[0] - x1[0]
        dy = x2[1] - x1[1]
        # self.logger.debug("x1: " + str(x1) + " x2: " + str(x2))
        # self.logger.debug("dxdy: %d, %d" %(dx,dy))
        if dy != 0:
            ret = np.abs(np.arctan(dx / dy))
        else:
            ret = np.pi / 2

        # sort out the sin
        sign = 1
        if dx < 0:
            sign = -1
        if dy > 0: # dy is opposite to normal
            ret = np.pi - np.abs(ret)
        return ret * sign


class AgentQ:
    def __init__(self, action_space, sensors_n):
        self.n_sensors = sensors_n

        obs_space = 2 ** sensors_n
        self.q_table = np.zeros((obs_space, action_space))

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.005

        self.step_counter = 0

    def get_action(self, observation):
        #observation is the sensor data 
        action_space = 3

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:            # print(q_table_avg)
            obs_n = self._convert_obs(observation)
            action_slice = self.q_table[0, :]
            action = np.argmax(action_slice) # select row for argmax
        else:
            action = random.randint(0, 2)

        return action # should be a number from 0-2

    def _convert_obs(self, observation):
        # converts from sensor 1 or 0 to a state number
        # 1101 --> 13
        obs_n = 0
        for i in range(len(observation)-1): #last sense doesn't work
            obs_n += observation[i] * (2**i)
        return int(obs_n)

    def update_q_table(self, obs, action, reward, next_obs):
        obs_n = self._convert_obs(obs)
        next_obs_n = self._convert_obs(next_obs)
        action_slice = self.q_table[next_obs_n,:]
        update_val = self.q_table[obs_n, action] * (1-self.learning_rate) + \
            self.learning_rate * (reward + self.discount_rate * np.max(action_slice))
        # I am changing this to max, not arg max

        self.q_table[obs_n, action] = update_val

        self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                np.exp(-self.exploration_decay_rate * self.step_counter)
        self.step_counter += 1

        # update exploration decay rate
        # code to come here when move away from greedy

    def save_q_table(self):
        file_location = 'Documents/ToyTrackProblem/'
        np.save(file_location + 'agent_q_table.npy', self.q_table)
        
        self.exploration_rate = self.min_exploration_rate

    def load_q_table(self):
        print("Loaded Q table")
        self.q_table = np.load('agent_q_table.npy')




