import numpy as np
import random

class AgentQ:
    def __init__(self, action_space, sensors_n):
        self.n_sensors = sensors_n

        obs_space = 2 ** sensors_n
        self.q_table = np.zeros((obs_space, action_space))

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.005

        self.step_counter = 0

    def get_action(self, observation):
        #observation is the sensor data 
        action_space = 3

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:            # print(q_table_avg)
            obs_n = self._convert_obs(observation)
            action_slice = self.q_table[obs_n, :]
            action = np.argmax(action_slice) # select row for argmax
        else:
            action = random.randint(0, 2)

        return action # should be a number from 0-2

    def _convert_obs(self, observation):
        # converts from sensor 1 or 0 to a state number
        # 1101 --> 13
        obs_n = 0
        for i in range(len(observation)-1): #last sense doesn't work
            obs_n += observation[i] * (2**i)
        return int(obs_n)

    def update_q_table(self, obs, action, reward, next_obs):
        obs_n = self._convert_obs(obs)
        next_obs_n = self._convert_obs(next_obs)
        q_s_prime = self.q_table[next_obs_n,:]
        update_val = self.q_table[obs_n, action] * (1-self.learning_rate) + \
            self.learning_rate * (reward + self.discount_rate * np.max(q_s_prime))

        self.q_table[obs_n, action] = update_val

        self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                np.exp(-self.exploration_decay_rate * self.step_counter)
        self.step_counter += 1

    def save_q_table(self):
        file_location = 'Documents/ToyTrackProblem/'
        np.save(file_location + 'agent_q_table.npy', self.q_table)
        
        self.exploration_rate = self.min_exploration_rate

    def load_q_table(self):
        print("Loaded Q table")
        self.q_table = np.load('agent_q_table.npy')

    def print_q(self):
        q = np.around(self.q_table, 3)
        print(q)


