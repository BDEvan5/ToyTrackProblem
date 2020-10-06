import numpy as np 
import casadi as ca 
from matplotlib import pyplot as plt


import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory

from rockit import *
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr



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

    def act(self, obs):
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

        # ds = self.deltas[self.pind:self.pind+1]
        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)
        # max_d = max(abs(ds))

        max_friction_force = 3.74 * 9.81 * 0.523 *0.9
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


class AgentMPC:
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

        self.ocp = None

        self.waypoints = None
        self.waypoint_last = None
        self.X_0 = None

        self.x = None
        self.y = None
        self.theta = None
        self.V = None
        self.delta = None

        self.current_d = None
        self.current_v = None

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

    def init_ocp(self):
        Nsim    = 30            # how much samples to simulate
        L       = 0.33             # bicycle model length
        nx      = 3             # the system is composed of 3 states
        nu      = 2             # the system has 2 control inputs
        N       = 10            # number of control intervals

        # -------------------------------
        # Set OCP
        # -------------------------------

        ocp = Ocp(T=FreeTime(10.0))

        # Define states
        self.x     = ocp.state()
        self.y     = ocp.state()
        self.theta = ocp.state()

        # Defince controls
        self.delta = ocp.control()
        self.V     = ocp.control(order=0)

        # Specify ODE
        ocp.set_der(self.x,      self.V*cos(self.theta))
        ocp.set_der(self.y,      self.V*sin(self.theta))
        ocp.set_der(self.theta,  self.V/L*tan(self.delta))

        # Define parameter
        self.X_0 = ocp.parameter(nx)

        # Initial constraints
        X = vertcat(self.x, self.y, self.theta)
        ocp.subject_to(ocp.at_t0(X) == self.X_0)

        # todo: change this to correct init
        # Initial guess
        ocp.set_initial(self.x,      0)
        ocp.set_initial(self.y,      0)
        ocp.set_initial(self.theta,  0)

        ocp.set_initial(self.V,    0.5)

        # Path constraints
        ocp.subject_to( 0 <= (self.V <= 1) )
        #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
        ocp.subject_to( -pi/6 <= (self.delta <= pi/6) )

        # Define physical path parameter
        self.waypoints = ocp.parameter(2, grid='control')
        self.waypoint_last = ocp.parameter(2)
        p = vertcat(self.x,self.y)

        ocp.add_objective(ocp.sum(sumsqr(p-self.waypoints), grid='control'))
        ocp.add_objective(sumsqr(ocp.at_tf(p)-self.waypoint_last))

        # Pick a solution method
        options = {"ipopt": {"print_level": 0}}
        options["expand"] = True
        options["print_time"] = False
        ocp.solver('ipopt', options)

        # Make it concrete for this ocp
        ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

        self.ocp = ocp

    def run_first_solve(self, x0, y0):
        ocp = self.ocp 

        wpts = self.get_current_wpts(x0, y0).T

        ocp.set_value(self.waypoints, wpts[:, :])
        ocp.set_value(self.waypoint_last, wpts[:, -1])

        current_X = vertcat(x0, y0, 0)
        ocp.set_value(self.X_0, current_X)

        # Solve the optimization problem
        sol = ocp.solve()

        # Get discretised dynamics as CasADi function to simulate the system
        Sim_system_dyn = ocp._method.discrete_system(ocp)

        t_sol, x_sol     = sol.sample(self.x,     grid='control')
        t_sol, y_sol     = sol.sample(self.y,     grid='control')
        t_sol, theta_sol = sol.sample(self.theta, grid='control')
        t_sol, delta_sol = sol.sample(self.delta, grid='control')
        t_sol, V_sol     = sol.sample(self.V,     grid='control')


        return [V_sol[0], delta_sol[0]]



    def get_current_wpts(self, x, y):
        N = 10

        self.pind = self.env_map.find_nearest_point([x, y])

        if self.pind + N < len(self.wpts):
            return self.wpts[self.pind: self.pind + N]
        
        n = len(self.wpts) - self.pind
        ret = self.wpts[self.pind:self.pind+n]
        for i in range(N-n):
            ret = np.append(ret, self.wpts[-1])
        
        return ret


    def mpc_act(self, obs):
        x, y, th = obs[0:3]

        current_X = vertcat(x, y, th)
        self.ocp.set_value(self.X_0, current_X)

        wpts = self.get_current_wpts(x, y).T

        self.ocp.set_value(self.waypoints, wpts[:, :])
        self.ocp.set_value(self.waypoint_last, wpts[:, -1])

        sol = self.ocp.solve()

        t_sol, x_sol     = sol.sample(self.x,     grid='control')
        t_sol, y_sol     = sol.sample(self.y,     grid='control')
        t_sol, theta_sol = sol.sample(self.theta, grid='control')
        t_sol, delta_sol = sol.sample(self.delta, grid='control')
        t_sol, V_sol     = sol.sample(self.V,     grid='control')

        tracking_error = sol.value(self.ocp.objective)
        print('Tracking error f', tracking_error)

        self.ocp.set_initial(self.x, x_sol)
        self.ocp.set_initial(self.y, y_sol)
        self.ocp.set_initial(self.theta, theta_sol)
        self.ocp.set_initial(self.delta, delta_sol)
        self.ocp.set_initial(self.V, V_sol)

        return [V_sol[0], delta_sol[0]]


    def act(self, obs):
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

        # ds = self.deltas[self.pind:self.pind+1]
        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)
        # max_d = max(abs(ds))

        max_friction_force = 3.74 * 9.81 * 0.523 *0.9
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





