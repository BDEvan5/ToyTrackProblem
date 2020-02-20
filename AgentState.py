import numpy as np 

class AgentState:
    # this represents the current state of the agent at any step
    def __init__(self):
        self.episode = 0
        self.step = 0
        self.reward = 0

        self.action = None
