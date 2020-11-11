import numpy as np 
import casadi as ca 
from matplotlib import pyplot as plt


import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory, generate_velocities
from TrackSimulator import ScanSimulator


class OptimalAgent:
    def __init__(self):
        self.name = "Optimal Agent: Following target references"
        self.env_map = None
        self.path_name = None
        self.wpts = None

        self.pind = 1
        self.target = None
        self.steps = 0

        self.current_v_ref = None
        self.current_phi_ref = None

        # self.scan_sim = ScanSimulator(20)

    def init_agent(self, env_map):
        self.env_map = env_map

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
 
        self.wpts = self.env_map.get_optimal_path()
        # self.wpts = self.env_map.get_reference_path()

        r_line = self.wpts
        ths = [lib.get_bearing(r_line[i], r_line[i+1]) for i in range(len(r_line)-1)]
        alphas = [lib.sub_angles_complex(ths[i+1], ths[i]) for i in range(len(ths)-1)]
        lds = [lib.get_distance(r_line[i], r_line[i+1]) for i in range(1, len(r_line)-1)]

        self.deltas = np.arctan(2*0.33*np.sin(alphas)/lds)

        self.pind = 1

        return self.wpts

    def act(self, obs):
        # scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])

        v_ref, d_ref = self.get_target_references(obs)

        # possibly clip if needed, but shouldn't change much.

        return [v_ref, d_ref]

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

        target = self.wpts[self.pind]
        th_target = lib.get_bearing(obs[0:2], target)
        alpha = lib.sub_angles_complex(th_target, obs[2])

        # pure pursuit
        ld = lib.get_distance(obs[0:2], target)
        delta_ref = np.arctan(2*0.33*np.sin(alpha)/ld)

        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)

        max_friction_force = 3.74 * 9.81 * 0.523 *0.6
        d_plan = max(abs(delta_ref), abs(obs[4]), max_d)
        theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
        v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
        v_ref = min(v_ref, 8.5)
        # v_ref = 3

        return v_ref, delta_ref

    def control_system(self, obs):
        v_ref = self.current_v_ref
        d_ref = self.current_phi_ref

        kp_a = 10
        a = (v_ref - obs[3]) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - obs[4]) * kp_delta

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 1
        while dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
                dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
            else:
                self.pind = 0

    def reset_lap(self):
        # for testing
        pass    

    def show_vehicle_history(self):
        pass

