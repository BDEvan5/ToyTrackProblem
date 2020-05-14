import numpy as np
from PathPlanner import A_StarPathFinder, WayPoint
from PathOptimisation import optmise_track_path, add_velocity, convert_to_obj
import LibFunctions as f


class AgentWrapper:
    # this agent is what will interface with the env.
    # this holds the global plan as used by the classical and RL agents
    def __init__(self, classical, rl, env):
        # self.rl = Vanilla()
        self.rl = rl
        self.classic = classical 
        self.env = env

        self.path = None

    def take_step(self, state):

        # rl_action = self.rl.get_action(state)
        rl_action = 1
        classical_action = self.classic.get_action(state)

        # action = self.add_actions(rl_action, classical_action)
        print(f"Classical Actin: {classical_action}")
        return classical_action

    def add_actions(self, rl, classic):
        action = rl + classic

        return action

        # theta_swerve = 0.8
        # # interpret action
        # # 0-m : left 
        # # m - straight
        # # m-n : right
        # agent_action += 1 # takes start from zero to 1

        # n_actions_side = (self.action_space -1)/2
        # m = n_actions_side + 1

        # if agent_action < m: # swerve left
        #     swerve = agent_action / n_actions_side * theta_swerve
        #     action = [con_action[0], con_action[1] -  swerve]
        #     dr = 1
        #     # print("Swerving left")
        # elif agent_action == m: # stay in the centre
        #     dr = 0
        #     swerve = 0
        #     action = con_action
        # elif agent_action > m: # swerve right
        #     swerve = (agent_action - m) / n_actions_side * theta_swerve
        #     action = [con_action[0], con_action[1] + swerve]
        #     dr = 1
        #     # print("Swerving right")
        # # print(swerve)
        # return action, dr
    
    def run_sim(self):
        env = self.env
        state, done, rewards = env.reset(), False, 0.0
        self.classic.reset()
        while not done:
            action = self.take_step(state)
            state, reward, done = env.step(action)
            rewards += reward
        print(f"EP complete, reward: {rewards}")

        return rewards

    def get_path_plan(self):
        track, car = self.env.track, self.classic.car
        path_finder = A_StarPathFinder(track)
        path = path_finder.run_search(5)
        path = optmise_track_path(path, track)
        path_obj = convert_to_obj(path)
        path_obj = add_velocity(path_obj, car)

        self.path = path_obj
        self.classic.path = path_obj

        # path_obj.show(track)




    



class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction

    def __call__(self, state, destination):
        # print(glbl_wp.x)
        # print(state.x)
        x_ref = destination.x 
        v_ref = destination.v 
        th_ref = destination.theta

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
        k = 0.25
        friction_c = 1
        return k * e_v + friction_c

    def _th_controll(self, e_th):
        # theta controller to come here when dth!= th
        # e_th = min(np.pi/2, e_th)
        # e_th = max(-np.pi/2, e_th) # clips the action to pi 

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


class Tracker:
    def __init__(self, path):
        self.path = path.route
        self.n_inds = len(self.path) -1

        self.pind = 0

        self.control_system = ControlSystem()

    def act(self, state):
        HORIZON = 1
        location = state.x

        ind = self.get_nearest_ind(location)
        if ind > self.pind: # pind is where I am now and want to work forward from.
            self.pind = ind

        ind = min(ind + HORIZON, self.n_inds) # checks that it isn't past the end indicie  
        destination = self.path[ind]

        destination.print_point(f"Destination: {ind}")

        ref_action = self.control_system(state, destination)

        return ref_action

    def get_nearest_ind(self, location):
        SEARCH_PTS = 5
        d = [f.get_distance(location, wpt.x) for wpt in self.path[self.pind:(self.pind+SEARCH_PTS)]]

        pt = min(d)
        ind = d.index(pt) + self.pind

        return ind







