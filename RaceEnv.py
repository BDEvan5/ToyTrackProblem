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
from StateStructs import SimMem, CarState, EnvState, SimulationState, WayPoint
from Interface import Interface
from Models import CarModel, TrackData


class RaceEnv:
    def __init__(self, config, track):
        self.config = config

        # set up and create
        logging.basicConfig(filename="AgentLogger.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = track
        self.car = CarModel()

        # memory structures
        self.car_state = CarState(self.config.ranges_n)
        self.env_state = EnvState()
        self.sim_mem = SimMem(self.logger)

    def step(self, control_action):

        new_x = self.car.chech_new_state(self.car_state, control_action, self.config.dt)
        coll_flag = self.track._check_collision_hidden(new_x)

        if not coll_flag: # no collsions
            self.car.update_controlled_state(self.car_state, control_action, self.config.dt)

        self.env_state.done = self._check_done(coll_flag)
        self.env_state.control_action = control_action
        self._get_reward(coll_flag)
        self._update_ranges()

        self.sim_mem.add_step(self.car_state, self.env_state)
        obs = self.car_state.get_state_observation()
        self.sim_mem.step += 1
        loc_state = self.car_state.x
        return self.car_state, self.env_state.reward, self.env_state.done

    def reset(self):
        self.car_state.reset_state(self.track.start_location, self.track.end_location)
        self.reward = 0
        self.sim_mem.clear_mem()
        self.track.set_up_random_obstacles() # turn back on

        # return self.car_state.get_state_observation()
        return self.car_state

    def _get_reward(self, coll_flag):
        self.car_state.cur_distance = f.get_distance(self.car_state.x, self.track.end_location) 

        reward = 0 
        crash_cost = 1

        if coll_flag:
            reward = - crash_cost
        
        self.env_state.reward = reward
        self.car_state.prev_distance = deepcopy(self.car_state.cur_distance)

    def _check_done(self, coll_flag):
        # check colision is end
        if coll_flag:
            # print("Ended in collision")
            return True

        if self.car_state.x[1] < 2 + self.config.dx:
            print("Destination reached")
            return True # this makes it a finish line not a point
        
        return False

    def _get_next_state(self, action):
        # this takes a direction to move in and moves there
        new_x = f.add_locations(action, self.car_state.x)
        return new_x

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

    def render_episode(self, screen_name_path, pause=False):
        dt = 100
        self.interface = Interface(self.track, dt)
        self.interface.save_shot_path = screen_name_path
        self.interface.pause_flag = pause

        for step in self.sim_mem.steps:
            self.interface.step_q.put(step)

        self.interface.setup_root()





