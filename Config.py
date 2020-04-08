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

        # set training
        self.network_update = 5
        self.render = True # dont show interface
        self.render_rate = 20
        self.test_rate = 10

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

        # model parameters
        self.dx = 8 # distance it must be from final target
        self.dt = 1 # controller frequency

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
                            0.99,
                            0.999,
                            0.995,
                            0.05,
                            5,
                            9,
                            3)
    
    return config

if __name__ == "__main__":
    c = create_sim_config()


