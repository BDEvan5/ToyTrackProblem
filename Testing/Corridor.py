import numpy as np 
import LibFunctions as lib

class CorridorAgent:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "Corridor"

    def act(self, obs):
        # all this does is go in the direction of the biggest range finder
        ranges = obs[2:]
        action = np.argmax(ranges)

        return action

    def load(self):
        pass



class PurePursuit:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space
        self.name = "PurePursuit"

    def load(self):
        pass

    def act(self, obs):
        try:
            grad = obs[1] / obs[0] # y/x
        except:
            grad = 10000
        angle = np.arctan(grad)
        if angle > 0:
            angle = np.pi - angle
        else:
            angle = - angle
        dth = np.pi / (self.act_space - 1)
        action = int(angle / dth)

        return action
