import numpy as np 
import casadi as ca 
from matplotlib import pyplot as plt


import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory, generate_velocities



class OptimalAgent:
    def __init__(self, env_map):
        self.env_map = env_map
        self.wpts = None
        self.vpts = None

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
        self.pind = 1
        self.target = None

    def init_agent(self):
        track = self.env_map.track
        n_set = MinCurvatureTrajectory(track, self.env_map.obs_map)

        deviation = np.array([track[:, 2] * n_set[:, 0], track[:, 3] * n_set[:, 0]]).T
        r_line = track[:, 0:2] + deviation

        self.wpts = r_line
        self.vpts = generate_velocities(r_line)
        # self.wpts = r_line

        self.pind = 1

        return self.wpts
        
    
    def act(self, obs):
        # v_ref, d_ref = self.get_corridor_references(obs)
        v_ref, d_ref = self.get_target_references(obs)
        a, d_dot = self.control_system(obs, v_ref, d_ref)

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return [a, d_dot]

    def get_corridor_references(self, obs):
        ranges = obs[5:]
        max_range = np.argmax(ranges)
        dth = np.pi / 9
        theta_dot = dth * max_range - np.pi/2

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        v_ref = 6

        return v_ref, delta_ref

    def get_target_references(self, obs):
        self._set_target(obs)

        # v_ref = 6
        v_ref = self.vpts[self.pind]

        th_target = lib.get_bearing(obs[0:2], self.wpts[self.pind])
        theta_dot = lib.sub_angles_complex(th_target, obs[2])
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
        if dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
            else:
                self.pind = 0

        
        # self.target = self.wpts[self.pind]



