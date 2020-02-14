import numpy as np 
import TrackEnv1


class Agent:
    # this class controlls the car
    def __init__(self, env):
        self.env = env

    def random_agent(self, steps=200):
        print("Random agent called")
        action_range = 1

        ep_reward = 0
        next_state = self.env.reset() # not needed until learning starts
        for t in range(steps):
            action = [np.random.randn(), np.random.randn()]
            # print(action)

            next_state, reward, done = self.env.step(action)
            # print(next_state.x)
            # print(reward)
            self.env.render() 

            # ep_reward += reward

            if done:
                break
        print("Episode done")
        