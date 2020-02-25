from copy import deepcopy
import logging


class EpisodeMem:
    def __init__(self, logger):
        self.logger = logger
        self.steps = []

    def add_step(self, state, action, reward):
        curr_step = StepInfo()
        copy_state = deepcopy(state)
        copy_action = deepcopy(action)
        curr_step.add_info(copy_state, copy_action, reward, len(self.steps))
        self.log_step(curr_step)
        self.steps.append(curr_step)

    def log_step(self, step):
        msg0 =  str(step.step) + ": ----------------------" + str(step.step)
        msg1 = "State: " + str(step.state.x) 
        msg2 = "Action: " + str(step.action)
        msg3 = "Reward: " + str(step.reward)

        self.logger.debug(msg0)
        self.logger.debug(msg1)
        self.logger.debug(msg2)
        self.logger.debug(msg3)

    def print_ep(self):
        for step in self.steps:
            step.print_step()
    

class StepInfo:
    def __init__(self):
        self.state = None
        self.reward = None
        self.action = None
        self.step = 0

    def print_step(self):
        msg0 = str(self.step)
        msg1 = " State: " + str(self.state.x) 
        msg2 = " Action: " + str(self.action)
        msg3 = " Reward: " + str(self.reward)

        print(msg0 + msg1 + msg2 + msg3)

    def add_info(self, state, action, reward, step=0):
        self.state = state
        self.reward = reward
        self.action = action
        self.step = step


