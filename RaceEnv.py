import matplotlib.pyplot as plt
import numpy as np
import time
import StateStructs as ls
from copy import deepcopy
import LibFunctions as f
import logging
from copy import deepcopy
import pickle
import datetime
import os
import PathPlanner
import SimulationMem as SimMem

class RaceEnv:
    def __init__(self, track, car, logger, dx=5, sense_dis=10):
        self.dx = dx # this is how close it must be to stop
        self.ds = sense_dis # this is how far the sensor can look ahead
        self.logger = logger
        self.track = track
        self.dt = 1 # update frequency

        self.state_space = 5
        self.action_space = 3

        self.car_state = ls.CarState(self.state_space)
        self.car_state.get_sense_observation
        self.env_state = ls.EnvState()
        self.car = car
        self.sim_mem = SimMem.SimMem(self.logger)
        self.c_sys = ControlSystem()
        self.wp = ls.WayPoint()
        self.wp_n = 1

        self.planner = PathPlanner.PathPlanner(track, car, logger)

    def step(self, agent_action):
        wp = self.track.route[self.wp_n]
        control_action = self.c_sys.get_controlled_action(self.car_state, wp)
        action, dr = self.get_new_action(agent_action, control_action)
        # print("Control step: " + str(control_action))
        # print("agent: " + str(agent_action))
        # print("Action: "  + str(action))
        new_x = self.car.chech_new_state(self.car_state, action, self.dt)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            self.car.update_controlled_state(self.car_state, action, self.dt)

        self._update_senses()
        self.env_state.done = self._check_done(coll_flag)
        self._get_reward(coll_flag, dr)
        self._update_ranges()
        self.env_state.action = action

        self.sim_mem.add_step(self.car_state, self.env_state)

        return self.car_state, self.env_state.reward, self.env_state.done

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
        self.car_state.x = deepcopy(self.track.start_location)
        self.car_state.v = 0
        self.car_state.theta = 0
        self._update_senses()
        self.reward = 0
        self.sim_mem.steps.clear()
        self.wp_n = 1
        self.planner.get_single_path()
        return self.car_state

    def _get_reward(self, coll_flag, dr):
        dis = f.get_distance(self.car_state.x, self.track.end_location) 

        reward = 0 # reward increases as distance decreases
        
        beta = 0.05
        swerve_cost = 1
        crash_cost = 100

        if coll_flag:
            reward = - crash_cost
        elif self.env_state.done:
            reward = 50
        else:
            reward = (100 - dis) * beta

        reward += -dr * swerve_cost

        self.env_state.distance_to_target = dis
        self.env_state.reward = reward

    def _check_done(self, coll_flag):
        # check colision is end
        if coll_flag:
            print("Ended in collision")
            return True

        # if no collision, check end
        dis = f.get_distance(self.track.end_location, self.car_state.x)
        if dis < self.dx:
            print("Final distance is: %d" % dis)
            return True
        
        # if not end, then update wp if needed
        wp = self.track.route[self.wp_n]
        wp_dis = f.get_distance(wp.x, self.car_state.x)
        if wp_dis < (self.dx/2):
            # print("Updating Wp")
            self.wp_n += 1

        return False

    def _update_senses(self):
        self.car_state.set_sense_locations(self.ds)
        b = self.track.boundary
        # self.car_state.print_sense()
        for sense in self.car_state.senses:
            if b[0] < sense.sense_location[0] < b[2]:
                if b[1] < sense.sense_location[1] < b[3]:
                    sense.val = 0
                    # It is within the boundary
                else:
                    # hits te wall boundary
                    sense.val = 1

            for o in self.track.obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
                        # if it hits an obstacle   
            for o in self.track.hidden_obstacles:
                if o[0] < sense.sense_location[0] < o[2]:
                    if o[1] < sense.sense_location[1] < o[3]:
                        sense.val = 1
        # self.car_state.print_sense()

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
            ran.val = (i - 2) * dx # sets the last distance before collision 
        # self.car_state.print_ranges()


class ControlSystem:
    def __init__(self):
        self.k_th_ref = 0.1 # amount to favour v direction

    def get_controlled_action(self, state, wp):
        # print(wp.x + state.x)
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


