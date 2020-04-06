import numpy as np 


class Configuration:
    def __init__(self, memory_size,
                    batch_size,
                    lr,
                    gamma,
                    eps,
                    eps_decay,
                    min_eps,
                    ranges_n,
                    state_space, 
                    action_space):

        # replay buffer parameters
        self.memory_size = memory_size
        self.batch_size = batch_size

        # learning parameters
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps

        self.value_c = 0.5
        self.entropy_c = 1e-4

        #env parameters
        self.ranges_n = ranges_n
        self.state_space = state_space
        self.action_space = action_space
        self.dx = 1 # distance it must be from final target

        # agent parameters
        self.max_steps = 200

    def step_eps(self):
        # call to update eps
        if self.eps > self.min_eps:
            self.eps *= self.eps_decay


def create_sim_config():
    config = Configuration(2000,
                            32,
                            7e-3,
                            0.999,
                            0.995,
                            0.05,
                            5,
                            7,
                            3)
    
    return config

if __name__ == "__main__":
    c = create_sim_config()


