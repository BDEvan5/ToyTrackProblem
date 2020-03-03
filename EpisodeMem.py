from copy import deepcopy
import logging
import LocationState as ls



class SimMem:
    def __init__(self, logger):
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
        msg1 = "State: x->" + str(step.x) + "v-> [" + str(step.v) + "] theta->" + str(step.theta)
        msg2 = "Action: " + str(step.action)
        # msg3 = "Reward: " + str(step.reward)

        self.logger.debug(msg0)
        self.logger.debug(msg1)
        self.logger.debug(msg2)
        # self.logger.debug(msg3)

    def print_ep(self):
        for i, step in enumerate(self.steps):
            step.print_step(i)