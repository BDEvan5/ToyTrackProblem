import LocationState as ls
import numpy as np
import LibFunctions as f
import EpisodeMem as em
import TrackEnv1


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
            self.control_to_wp(w_pt)

    def control_to_wp(self, wp):
        i = 0
        max_steps = 100

        while not self.at_target(wp) and i < max_steps: #keeps going till at each pt
            
            action = self.control_sys.get_controlled_action(self.state, wp)
            self.state, done = self.env.control_step(action)

            if done:
                break

            # self.logger.debug("Current Target: " + str(wp.x) + "V_th" + str(wp.v) + "@" + str(wp.theta))
            # self.logger.debug("")
            i+=1 

    def at_target(self, wp):
        way_point_dis = f.get_distance(self.state.x, wp.x)
        # print(way_point_dis)
        if way_point_dis < 2:
            print("Way point reached: " + str(wp.x))
            return True
        return False
    
    def check_action_rl(self, action):
        # write code here to check if the rl wants to act.   
        pass     


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

    def get_action(self, observation):
        #observation is the sensor data 
        obs_n = self._convert_obs(observation)
        action = np.argmax(self.q_table[obs_n,:]) # select row for argmax
        return action # should be a number from 0-2

    def _convert_obs(self, observation):
        # converts from sensor 1 or 0 to a state number
        # 1101 --> 13
        obs_n = 0
        for i in range(observation):
            obs_n += observation * (2**i)
        return obs_n

    def update_q_table(self, obs, action, reward, next_obs):
        obs_n = self._convert_obs(obs)
        next_obs_n = self._convert_obs(next_obs)

        update_val = self.q_table[obs_n, action] * (1-self.learning_rate) + \
            self.learning_rate * (reward + self.discount_rate * np.argmax(self.q_table[next_obs_n, :]))

        self.q_table[obs_n, action] = update_val

        # update exploration decay rate
        # code to come here when move away from greedy
