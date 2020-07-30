import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CommonTestUtils import PurePursuit, single_rep_eval
from DQN_PureRep import Qnet
from DQN_Switch import ValueNet

      

class TestSwitchRep:
    def __init__(self, obs_space, action_space, switch_name, rep_name):
        self.value_model = None
        self.model = None
        self.pp = PurePursuit(obs_space, action_space)

        self.action_space = action_space
        self.obs_space = obs_space

        self.switch_name = switch_name
        self.rep_name = rep_name

    def load(self, directory="./dqn_saves"):
        filename = '%s/%s_Vmodel.pth' % (directory, self.switch_name)
        self.value_model = torch.load(filename)
        filename = self.rep_name
        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        print(f"Loaded Agent: {self.switch_name}")

    def try_load(self, load=True):
        if load:
            try:
                self.load()
            except:
                print(f"Unable to load model")
                raise LookupError
                
    def decide(self, obs, pp_action):
        value_obs = np.append(obs, pp_action)
        value_obs_t = torch.from_numpy(value_obs).float()
        safe_value = self.value_model.forward(value_obs_t)

        return safe_value.detach().item() 

    def rep_act(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        action = out.argmax().item()
        return action

    def act(self, obs):
        pp_action = self.pp.act(obs)
        safe_value = self.decide(obs, pp_action)

        if safe_value > 0: # threshold to be decided. 
            action = pp_action
            return action 

        action = self.rep_act(obs)
        return action
    
if __name__ == "__main__":
    switch_name = "SwitchSR"
    rep_name = "RepSW"

    agent = TestSwitchRep(12, 10, switch_name, rep_name)
    agent.load()
    single_rep_eval(agent, True)







