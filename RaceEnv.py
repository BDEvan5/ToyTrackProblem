import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy
import pickle
import datetime
import os
from PathPlanner import PathPlanner
from StateStructs import SimMem, CarState, EnvState, SimulationState, WayPoint
from Interface import Interface
from Models import CarModel, TrackData

class RaceEnv:
    def __init__(self, config):
        self.config = config

        # set up and create
        logging.basicConfig(filename="AgentLogger.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = TrackData()
        self.car = CarModel()
        self.planner = PathPlanner(self.track, self.car, self.logger)
        self.c_sys = ControlSystem()


        # configurable
        self.dt = 1 # Control system frequency 
        # TODO: replace with config dt

        # memory structures
        self.car_state = CarState(self.config.ranges_n)
        self.env_state = EnvState()
        self.sim_mem = SimMem(self.logger)
        self.glbl_wp_n = 0 # waypoint number in the list of global way points


    def step(self, agent_action):
        #TODO: move this to a get lcl_wp function call that returns lcl_wp
        # agent action is in form of th for set r = 10
        r = 10 # possibly make more flexible later on
        th = (agent_action - 1) * 0.6 # 0.6 radians to the right or left of current pos
        lcl_x = [ r * np.sin(th), -r * np.cos(th)]
        lcl_wp = WayPoint()
        lcl_wp.x = f.add_locations(self.car_state.x, lcl_x)
        # print(lcl_wp.x)
        lcl_wp.v = self.car.max_v # this will be changed later
        lcl_wp.theta = th # orientation of lcl_wp. It is as if you go straight through it

        control_action = self.c_sys.get_controlled_action(self.car_state, lcl_wp)
        # action, dr = self.get_new_action(agent_action, control_action)
        new_x = self.car.chech_new_state(self.car_state, control_action, self.config.dt)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            self.car.update_controlled_state(self.car_state, control_action, self.dt)

        self.env_state.done = self._check_done(coll_flag)
        self.env_state.control_action = control_action
        self.env_state.agent_action = agent_action
        self._get_reward(coll_flag, agent_action)
        self._update_ranges()

        self.sim_mem.add_step(self.car_state, self.env_state)
        obs = self.car_state.get_state_observation()
        self.sim_mem.step += 1
        return obs, self.env_state.reward, self.env_state.done

    def model_step(self, state, agent_action):
        glbl_wp = self.track.route[self.glbl_wp_n]
        control_action = self.c_sys.get_controlled_action(self.car_state, glbl_wp)
        action, dr = self.get_new_action(agent_action, control_action)
        new_x = self.car.chech_new_state(self.car_state, action, self.dt)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            state = self.car.update_modelled_state(self.car_state, action, self.dt)
        else:
            state = deepcopy(self.car_state)

        dx = 5 # search size
        curr_x = state.x
        th_car = state.theta
        for ran in state.ranges:
            th = th_car + ran.angle
            crash_val = 0
            i = 0
            while crash_val == 0:
                r = dx * i
                addx = [dx * i * np.sin(th), -dx * i * np.cos(th)] # check
                x_search = f.add_locations(curr_x, addx)
                crash_val = self.track._check_collision_hidden(x_search)
                i += 1
            update_val = (i - 2) * dx # sets the last distance before collision 
            ran.dr = update_val - ran.val # possibly take more than two terms
            ran.val = update_val

        obs = state.get_state_observation(control_action)

        return obs

    def get_new_action(self, agent_action, con_action):
        theta_swerve = 0.8
        # interpret action
        # 0-m : left 
        # m - straight
        # m-n : right
        agent_action += 1 # takes start from zero to 1

        n_actions_side = (self.action_space -1)/2
        m = n_actions_side + 1

        if agent_action < m: # swerve left
            swerve = agent_action / n_actions_side * theta_swerve
            action = [con_action[0], con_action[1] -  swerve]
            dr = 1
            # print("Swerving left")
        elif agent_action == m: # stay in the centre
            dr = 0
            swerve = 0
            action = con_action
        elif agent_action > m: # swerve right
            swerve = (agent_action - m) / n_actions_side * theta_swerve
            action = [con_action[0], con_action[1] + swerve]
            dr = 1
            # print("Swerving right")
        # print(swerve)
        return action, dr

    def reset(self):
        # resets to starting location
        self.car_state.prev_distance = f.get_distance(self.track.start_location, self.track.end_location)
        self.car_state.reset_state(self.track.start_location)
        self.planner.get_single_path() # plans the path to be followed
        self.glbl_wp_n = 0
        self.car_state.glbl_wp = self.track.route[self.glbl_wp_n] # sets current goal
        self.reward = 0
        self.sim_mem.steps.clear()
        self.sim_mem.step = 0
        self.track.set_up_random_obstacles()

        return self.car_state.get_state_observation()

    def _get_reward(self, coll_flag, agent_action):
        self.car_state.cur_distance = f.get_distance(self.car_state.x, self.track.end_location) 

        reward = 0 # reward increases as distance decreases
        
        beta1 = 0.08
        beta2 = 0.05
        beta3 = 1
        swerve_cost = 1
        crash_cost = 100
        goal_reward = 100

        if coll_flag:
            reward = - crash_cost
        elif self.env_state.done:
            reward = beta3 * ( goal_reward - self.sim_mem.step) # gives reward * number of steps
        else:
            ranges = [ran.val for ran in self.car_state.ranges]
            min_range = np.min(ranges)

            # reward = beta1 * self.car_state.get_distance_difference()
            reward = (min_range) * beta2

        reward += - np.abs(agent_action - 1) * swerve_cost

        self.env_state.reward = reward
        self.car_state.prev_distance = deepcopy(self.car_state.cur_distance)

    def _check_done(self, coll_flag):
        # check colision is end
        if coll_flag:
            # print("Ended in collision")
            return True

        # if no collision, check end
        end_dis = f.get_distance(self.track.end_location, self.car_state.x)
        if end_dis < self.config.dx:
            # print("Final distance is: %d" % end_dis)
            return True
        
        # if not end, then update glbl_wp if needed
        wp_dis = f.get_distance(self.car_state.glbl_wp.x, self.car_state.x)
        if wp_dis < (self.config.dx/2):
            self.glbl_wp_n += 1
            self.car_state.glbl_wp = self.track.route[self.glbl_wp_n]

        return False

    def _get_next_state(self, action):
        # this takes a direction to move in and moves there
        new_x = f.add_locations(action, self.car_state.x)
        return new_x

    def get_ep_mem(self):
        # self.sim_mem.print_ep()
        return self.sim_mem

    def _update_ranges(self):
        dx = 5 # search size
        curr_x = self.car_state.x
        th_car = self.car_state.theta
        for ran in self.car_state.ranges:
            th = th_car + ran.angle
            crash_val = 0
            i = 0
            while crash_val == 0:
                r = dx * i
                addx = [dx * i * np.sin(th), -dx * i * np.cos(th)] # check
                x_search = f.add_locations(curr_x, addx)
                crash_val = self.track._check_collision_hidden(x_search)
                i += 1
            update_val = (i - 2) * dx # sets the last distance before collision 
            ran.dr = update_val - ran.val # possibly take more than two terms
            ran.val = update_val
        # self.car_state.print_ranges()

    def render_episode(self, screen_name_path):
        dt = 30
        self.interface = Interface(self.track, dt)
        self.interface.save_shot_path = screen_name_path

        self.interface.pause_flag = False # starts in play mode
        for step in self.sim_mem.steps:
            self.interface.step_q.put(step)

        self.interface.setup_root()


class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction

    def get_controlled_action(self, state, glbl_wp):
        # print(glbl_wp.x)
        # print(state.x)
        x_ref = glbl_wp.x 
        v_ref = glbl_wp.v 
        th_ref = glbl_wp.theta

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


