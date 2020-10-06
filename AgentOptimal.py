import numpy as np 
import casadi as ca 
from matplotlib import pyplot as plt


import LibFunctions as lib
from RaceTrackMap import TrackMap

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


# class AgentMPC:
#     def __init__(self):
#         self.name = "Optimal Agent: Following target references"
#         self.env_map = None
#         self.path_name = None
#         self.wpts = None

#         self.pind = 1
#         self.target = None
#         self.steps = 0

#         self.current_v_ref = None
#         self.current_phi_ref = None

#         self.ocp = None

#         self.waypoints = None
#         self.waypoint_last = None
#         self.X_0 = None

#         self.x = None
#         self.y = None
#         self.theta = None
#         self.V = None
#         self.delta = None

#         self.current_d = None
#         self.current_v = None

#     def init_agent(self, env_map):
#         self.env_map = env_map
#         self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
 
#         self.wpts = self.env_map.get_min_curve_path()

#         r_line = self.wpts
#         ths = [lib.get_bearing(r_line[i], r_line[i+1]) for i in range(len(r_line)-1)]
#         alphas = [lib.sub_angles_complex(ths[i+1], ths[i]) for i in range(len(ths)-1)]
#         lds = [lib.get_distance(r_line[i], r_line[i+1]) for i in range(1, len(r_line)-1)]

#         self.deltas = np.arctan(2*0.33*np.sin(alphas)/lds)

#         self.pind = 1

#         return self.wpts

#     def init_ocp(self):
#         Nsim    = 30            # how much samples to simulate
#         L       = 0.33             # bicycle model length
#         nx      = 3             # the system is composed of 3 states
#         nu      = 2             # the system has 2 control inputs
#         N       = 10            # number of control intervals

#         # -------------------------------
#         # Set OCP
#         # -------------------------------

#         ocp = Ocp(T=FreeTime(10.0))

#         # Define states
#         self.x     = ocp.state()
#         self.y     = ocp.state()
#         self.theta = ocp.state()

#         # Defince controls
#         self.delta = ocp.control()
#         self.V     = ocp.control(order=0)

#         # Specify ODE
#         ocp.set_der(self.x,      self.V*cos(self.theta))
#         ocp.set_der(self.y,      self.V*sin(self.theta))
#         ocp.set_der(self.theta,  self.V/L*tan(self.delta))

#         # Define parameter
#         self.X_0 = ocp.parameter(nx)

#         # Initial constraints
#         X = vertcat(self.x, self.y, self.theta)
#         ocp.subject_to(ocp.at_t0(X) == self.X_0)

#         # todo: change this to correct init
#         # Initial guess
#         ocp.set_initial(self.x,      0)
#         ocp.set_initial(self.y,      0)
#         ocp.set_initial(self.theta,  0)

#         ocp.set_initial(self.V,    0.5)

#         # Path constraints
#         ocp.subject_to( 0 <= (self.V <= 1) )
#         #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
#         ocp.subject_to( -pi/6 <= (self.delta <= pi/6) )

#         # Define physical path parameter
#         self.waypoints = ocp.parameter(2, grid='control')
#         self.waypoint_last = ocp.parameter(2)
#         p = vertcat(self.x,self.y)

#         ocp.add_objective(ocp.sum(sumsqr(p-self.waypoints), grid='control'))
#         ocp.add_objective(sumsqr(ocp.at_tf(p)-self.waypoint_last))

#         # Pick a solution method
#         options = {"ipopt": {"print_level": 0}}
#         options["expand"] = True
#         options["print_time"] = False
#         ocp.solver('ipopt', options)

#         # Make it concrete for this ocp
#         ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

#         self.ocp = ocp

#     def run_first_solve(self, x0, y0):
#         ocp = self.ocp 

#         wpts = self.get_current_wpts(x0, y0).T

#         ocp.set_value(self.waypoints, wpts[:, :])
#         ocp.set_value(self.waypoint_last, wpts[:, -1])

#         current_X = vertcat(x0, y0, 0)
#         ocp.set_value(self.X_0, current_X)

#         # Solve the optimization problem
#         sol = ocp.solve()

#         # Get discretised dynamics as CasADi function to simulate the system
#         Sim_system_dyn = ocp._method.discrete_system(ocp)

#         t_sol, x_sol     = sol.sample(self.x,     grid='control')
#         t_sol, y_sol     = sol.sample(self.y,     grid='control')
#         t_sol, theta_sol = sol.sample(self.theta, grid='control')
#         t_sol, delta_sol = sol.sample(self.delta, grid='control')
#         t_sol, V_sol     = sol.sample(self.V,     grid='control')


#         return [V_sol[0], delta_sol[0]]



#     def get_current_wpts(self, x, y):
#         N = 10

#         self.pind = self.env_map.find_nearest_point([x, y])

#         if self.pind + N < len(self.wpts):
#             return self.wpts[self.pind: self.pind + N]
        
#         n = len(self.wpts) - self.pind
#         ret = self.wpts[self.pind:self.pind+n]
#         for i in range(N-n):
#             ret = np.append(ret, self.wpts[-1])
        
#         return ret


#     def mpc_act(self, obs):
#         x, y, th = obs[0:3]

#         current_X = vertcat(x, y, th)
#         self.ocp.set_value(self.X_0, current_X)

#         wpts = self.get_current_wpts(x, y).T

#         self.ocp.set_value(self.waypoints, wpts[:, :])
#         self.ocp.set_value(self.waypoint_last, wpts[:, -1])

#         sol = self.ocp.solve()

#         t_sol, x_sol     = sol.sample(self.x,     grid='control')
#         t_sol, y_sol     = sol.sample(self.y,     grid='control')
#         t_sol, theta_sol = sol.sample(self.theta, grid='control')
#         t_sol, delta_sol = sol.sample(self.delta, grid='control')
#         t_sol, V_sol     = sol.sample(self.V,     grid='control')

#         tracking_error = sol.value(self.ocp.objective)
#         print('Tracking error f', tracking_error)

#         self.ocp.set_initial(self.x, x_sol)
#         self.ocp.set_initial(self.y, y_sol)
#         self.ocp.set_initial(self.theta, theta_sol)
#         self.ocp.set_initial(self.delta, delta_sol)
#         self.ocp.set_initial(self.V, V_sol)

#         return [V_sol[0], delta_sol[0]]


#     def act(self, obs):
#         v_ref, d_ref = self.get_target_references(obs)

#         # possibly clip if needed, but shouldn't change much.

#         return [v_ref, d_ref]

#     def get_corridor_references(self, obs):
#         ranges = obs[5:]
#         max_range = np.argmax(ranges)
#         dth = np.pi / 9
#         theta_dot = dth * max_range - np.pi/2

#         L = 0.33
#         delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

#         v_ref = 6

#         return v_ref, delta_ref

#     def get_target_references(self, obs):
#         self._set_target(obs)

#         target = self.wpts[self.pind]
#         th_target = lib.get_bearing(obs[0:2], target)
#         alpha = lib.sub_angles_complex(th_target, obs[2])

#         # pure pursuit
#         ld = lib.get_distance(obs[0:2], target)
#         delta_ref = np.arctan(2*0.33*np.sin(alpha)/ld)

#         # ds = self.deltas[self.pind:self.pind+1]
#         ds = self.deltas[min(self.pind, len(self.deltas)-1)]
#         max_d = abs(ds)
#         # max_d = max(abs(ds))

#         max_friction_force = 3.74 * 9.81 * 0.523 *0.9
#         d_plan = max(abs(delta_ref), abs(obs[4]), max_d)
#         theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
#         v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
#         v_ref = min(v_ref, 8.5)
#         # v_ref = 3

#         return v_ref, delta_ref

#     def control_system(self, obs):
#         v_ref = self.current_v_ref
#         d_ref = self.current_phi_ref

#         kp_a = 10
#         a = (v_ref - obs[3]) * kp_a
        
#         kp_delta = 40
#         d_dot = (d_ref - obs[4]) * kp_delta

#         return a, d_dot

#     def _set_target(self, obs):
#         dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
#         shift_distance = 1
#         while dis_cur_target < shift_distance: # how close to say you were there
#             if self.pind < len(self.wpts)-2:
#                 self.pind += 1
#                 dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
#             else:
#                 self.pind = 0

#     def reset_lap(self):
#         # for testing
#         pass    


def find_closest_point(pose, reference_path, start_index):
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]

    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    print('find_closest_point results in', index_closest)
    return index_closest

def index_last_point_fun(start_index, wp, dist):
    pathpoints = wp.shape[1]
    cum_dist = 0
    for i in range(start_index, pathpoints-1):
        cum_dist += np.linalg.norm(wp[:,i] - wp[:,i+1])
        if cum_dist >= dist:
            return i + 1
    return pathpoints - 1

def get_current_waypoints(start_index, wp, N, dist):
    last_index = index_last_point_fun(start_index, wp, dist)
    delta_index = last_index - start_index
    if delta_index >= N:
        index_list = list(range(start_index, start_index+N+1))
        print('index list with >= N points:', index_list)
    else:
        index_list = list(range(start_index, last_index)) + [last_index]*(N-delta_index+1)
        print('index list with < N points:', index_list)
    return wp[:,index_list]


class AgentMPC:
    def __init__(self):
        self.env_map = None

        self.ocp = None

        self.Nsim    = 20            # how much samples to simulate
        self.L       = 0.03             # bicycle model length
        self.nx      = 3             # the system is composed of 3 states
        self.nu      = 2             # the system has 2 control inputs
        self.N       = 10            # number of control intervals

        self.time_hist      = np.zeros((self.Nsim+1, self.N+1))
        self.x_hist         = np.zeros((self.Nsim+1, self.N+1))
        self.y_hist         = np.zeros((self.Nsim+1, self.N+1))
        self.theta_hist     = np.zeros((self.Nsim+1, self.N+1))
        self.delta_hist     = np.zeros((self.Nsim+1, self.N+1))
        self.V_hist         = np.zeros((self.Nsim+1, self.N+1))

        self.tracking_error = np.zeros((self.Nsim+1, 1))

        self.x = None
        self.y = None
        self.theta = None
        self.V = None
        self.delta = None
        self.X_0 = None
        self.waypoint_last = None
        self.waypoints = None

        self.distance = None
        self.ref_path = None

    def init_agent(self, env_map):
        self.env_map = env_map

        self.init_path()

        self.init_ocp()

    def init_ocp(self):
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
        ocp.set_der(self.theta,  self.V/self.L*tan(self.delta))

        # Define parameter
        self.X_0 = ocp.parameter(self.nx)

        # Initial constraints
        X = vertcat(self.x, self.y, self.theta)
        ocp.subject_to(ocp.at_t0(X) == self.X_0)

        # Path constraints
        ocp.subject_to( 0 <= (self.V <= 1) )
        #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
        ocp.subject_to( -pi/6 <= (self.delta <= pi/6) )

        # Minimal time
        # ocp.add_objective(0.50*ocp.T)

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
        ocp.method(MultipleShooting(N=self.N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

        ocp.set_initial(self.x,      0)
        ocp.set_initial(self.y,      0)
        ocp.set_initial(self.theta,  0)

        ocp.set_initial(self.V,    0.5)

        self.ocp = ocp

    def init_path(self):
        wpts = self.env_map.get_min_curve_path()
        self.ref_path = {}
        mul = 1
        self.ref_path['x'] = wpts[:, 0] *mul
        self.ref_path['y'] = wpts[:, 1] *mul
        self.wp = horzcat(self.ref_path['x'], self.ref_path['y']).T

        self.distance = 5 * mul

    def first_solve(self):
        ocp = self.ocp

        # First waypoint is current position
        index_closest_point = 0

        # Create a list of N waypoints
        current_waypoints = get_current_waypoints(index_closest_point, self.wp, 
                                        self.N, dist=self.distance)

        # Set initial value for waypoint parameters
        ocp.set_value(self.waypoints, current_waypoints[:,:-1])
        ocp.set_value(self.waypoint_last, current_waypoints[:,-1])

        # Set initial value for states
        current_X = vertcat(self.ref_path['x'][0], self.ref_path['y'][0], 0)
        ocp.set_value(self.X_0, current_X)

        # Solve the optimization problem
        sol = ocp.solve()

        # Get discretised dynamics as CasADi function to simulate the system
        self.Sim_system_dyn = ocp._method.discrete_system(ocp)

        # Log data for post-processing
        t_sol, x_sol     = sol.sample(self.x,     grid='control')
        t_sol, y_sol     = sol.sample(self.y,     grid='control')
        t_sol, theta_sol = sol.sample(self.theta, grid='control')
        t_sol, delta_sol = sol.sample(self.delta, grid='control')
        t_sol, V_sol     = sol.sample(self.V,     grid='control')

        self.time_hist[0,:]    = t_sol
        self.x_hist[0,:]       = x_sol
        self.y_hist[0,:]       = y_sol
        self.theta_hist[0,:]   = theta_sol
        self.delta_hist[0,:]   = delta_sol
        self.V_hist[0,:]       = V_sol

        self.tracking_error[0] = sol.value(ocp.objective)

        return delta_sol, V_sol, current_X, t_sol

    def run_sim(self):
        ocp = self.ocp
        delta_sol, V_sol, current_X, t_sol = self.first_solve()
        index_closest_point = 0
        for i in range(self.Nsim):
            print("timestep", i+1, "of", self.Nsim)

            # Combine first control inputs
            current_U = vertcat(delta_sol[0], V_sol[0])

            # Simulate dynamics (applying the first control input) and update the current state
            current_X = self.Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]

            # Set the parameter X0 to the new current_X
            ocp.set_value(self.X_0, current_X)

            # Find closest point on the reference path compared witch current position
            index_closest_point = find_closest_point(current_X[:2], self.ref_path, index_closest_point)

            # Create a list of N waypoints
            current_waypoints = get_current_waypoints(index_closest_point, self.wp, 
                            self.N, dist=self.distance)

            # Set initial value for waypoint parameters
            ocp.set_value(self.waypoints, current_waypoints[:,:-1])
            ocp.set_value(self.waypoint_last, current_waypoints[:,-1])

            # Solve the optimization problem
            sol = ocp.solve()

            # Log data for post-processing
            t_sol, x_sol     = sol.sample(self.x,     grid='control')
            t_sol, y_sol     = sol.sample(self.y,     grid='control')
            t_sol, theta_sol = sol.sample(self.theta, grid='control')
            t_sol, delta_sol = sol.sample(self.delta, grid='control')
            t_sol, V_sol     = sol.sample(self.V,     grid='control')

            self.time_hist[i+1,:]    = t_sol
            self.x_hist[i+1,:]       = x_sol
            self.y_hist[i+1,:]       = y_sol
            self.theta_hist[i+1,:]   = theta_sol
            self.delta_hist[i+1,:]   = delta_sol
            self.V_hist[i+1,:]       = V_sol

            self.tracking_error[i+1] = sol.value(ocp.objective)
            print('Tracking error f', self.tracking_error[i+1])

            ocp.set_initial(self.x, x_sol)
            ocp.set_initial(self.y, y_sol)
            ocp.set_initial(self.theta, theta_sol)
            ocp.set_initial(self.delta, delta_sol)
            ocp.set_initial(self.V, V_sol)

    def plot_results(self):
        T_start = 0
        T_end = sum(self.time_hist[k,1] - self.time_hist[k,0] for k in range(self.Nsim+1))

        fig = plt.figure()

        ax2 = plt.subplot(2, 2, 1)
        ax3 = plt.subplot(2, 2, 2)
        ax4 = plt.subplot(2, 2, 3)
        ax5 = plt.subplot(2, 2, 4)

        ax2.plot(self.wp[0,:], self.wp[1,:], 'ko')
        ax2.set_xlabel('x pos [m]')
        ax2.set_ylabel('y pos [m]')
        ax2.set_aspect('equal', 'box')

        ax3.set_xlabel('T [s]')
        ax3.set_ylabel('pos [m]')
        ax3.set_xlim(0,T_end)

        ax4.axhline(y= pi/6, color='r')
        ax4.axhline(y=-pi/6, color='r')
        ax4.set_xlabel('T [s]')
        ax4.set_ylabel('delta [rad/s]')
        ax4.set_xlim(0,T_end)

        ax5.axhline(y=0, color='r')
        ax5.axhline(y=1, color='r')
        ax5.set_xlabel('T [s]')
        ax5.set_ylabel('V [m/s]')
        ax5.set_xlim(0,T_end)

        # fig2 = plt.figure()
        # ax6 = plt.subplot(1,1,1)

        for k in range(self.Nsim+1):
            # ax6.plot(time_hist[k,:], delta_hist[k,:])
            # ax6.axhline(y= pi/6, color='r')
            # ax6.axhline(y=-pi/6, color='r')

            ax2.plot(self.x_hist[k,:], self.y_hist[k,:], 'b-')
            ax2.plot(self.x_hist[k,:], self.y_hist[k,:], 'g.')
            ax2.plot(self.x_hist[k,0], self.y_hist[k,0], 'ro', markersize = 10)

            ax3.plot(T_start, self.x_hist[k,0], 'b.')
            ax3.plot(T_start, self.y_hist[k,0], 'r.')

            ax4.plot(T_start, self.delta_hist[k,0], 'b.')
            ax5.plot(T_start, self.V_hist[k,0],     'b.')

            T_start = T_start + (self.time_hist[k,1] - self.time_hist[k,0])
            plt.pause(0.05)

        ax3.legend(['x pos [m]','y pos [m]'])

        fig3 = plt.figure()
        ax1 = plt.subplot(1,1,1)
        ax1.semilogy(self.tracking_error)
        ax1.set_xlabel('N [-]')
        ax1.set_ylabel('obj f')

        plt.show(block=True)

def main():
    env_map = TrackMap()
    myMpc = AgentMPC()
    myMpc.init_agent(env_map)

    myMpc.run_sim()
    myMpc.plot_results()




if __name__ == "__main__":
    main()
