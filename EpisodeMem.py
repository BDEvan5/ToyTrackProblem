from copy import deepcopy
import logging
import LocationState as ls
import pickle
import datetime
import os


class SimMem:
    def __init__(self, logger=None):
        self.steps = []
        self.logger = logger

        self.step = 0

    def add_step(self, car_state, env_state):
        SimStep = ls.SimulationState()
        SimStep._add_car_state(deepcopy(car_state))
        SimStep._add_env_state(deepcopy(env_state))
        self.steps.append(SimStep)
        self.log_step(SimStep)
        self.step += 0

    def log_step(self, step):
        msg0 =  str(step.step) + ": ----------------------" + str(step.step)
        msg1 = "State: x->" + str(step.car_state.x) + "v-> [" + str(step.car_state.v) + "] theta->" + str(step.car_state.theta)
        msg2 = "Action: " + str(step.env_state.action)
        # msg3 = "Reward: " + str(step.reward)

        self.logger.debug(msg0)
        self.logger.debug(msg1)
        self.logger.debug(msg2)
        # self.logger.debug(msg3)

    def print_ep(self):
        for i, step in enumerate(self.steps):
            step.print_step(i)

    def save_ep(self, f_name="Last_ep"):
        save_file_name = "Documents/ToyTrackProblem/"  + f_name # + str(datetime.datetime.now())
        
        if os.path.exists(save_file_name):
            os.remove(save_file_name)
        
        s_file = open(save_file_name, 'ab')

        pickle.dump(self.steps, s_file)

        s_file.close()

    def load_ep(self, f_name="Last_ep"):
        save_file_name = "Documents/ToyTrackProblem/" + f_name
        s_file = open(save_file_name, 'rb')

        self.steps = pickle.load(s_file)

        s_file.close()