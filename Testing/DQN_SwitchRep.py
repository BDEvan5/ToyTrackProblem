import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CommonTestUtils import PurePursuit, single_rep_eval


class ValueNet(nn.Module):
    def __init__(self, obs_space):
        super(ValueNet, self).__init__()
        h_vals = int(h_size / 2)
        self.fc1 = nn.Linear(obs_space + 1, h_vals)
        self.fc2 = nn.Linear(h_vals, h_vals)
        self.fc3 = nn.Linear(h_vals, 1)
        self.exploration_rate = EXPLORATION_MAX
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Qnet(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Qnet, self).__init__()
        h_size = 512
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.exploration_rate = EXPLORATION_MAX

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if torch.any(torch.isnan(x)):
            print(f"Error in model values: nan detected")
            print(f"Obs: {obs}")
            print(f"Weights: {self.fc1.weight}")
            # raise ValueError
        return x
      

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







