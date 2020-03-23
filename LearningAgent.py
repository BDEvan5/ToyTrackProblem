import numpy as np
import random

class AgentQ_TD:
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


class AgentLamTD:
    def __init__(self, action_space, sensors_n):
        self.n_sensors = sensors_n

        self.obs_space = 2 ** (sensors_n)
        self.action_space = action_space
        self.q_table = np.zeros((self.obs_space, action_space))
        self.E = np.zeros((self.obs_space, action_space))

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.0009
        self.lam = 0.5

        self.step_counter = 0

    def get_action(self, state):
        # print("ag" + str(type(state)))
        observation = state.get_sense_observation()
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:            # print(q_table_avg)
            obs_n = self._convert_obs(observation)
            action_slice = self.q_table[obs_n, :]
            action = np.argmax(action_slice) # select row for argmax
        else:
            action = random.randint(0, self.action_space-1)

        return action # should be a number from 0-2

    def _convert_obs(self, observation):
        # converts from sensor 1 or 0 to a state number
        # 1101 --> 13
        obs_n = 0
        for i in range(len(observation)): 
            obs_n += observation[i] * (2**i)
        return int(obs_n)

    def update_q_table(self, state, action, reward, next_state): 
        obs = state.get_sense_observation()
        next_obs = next_state.get_sense_observation()

        obs_n = self._convert_obs(obs)
        next_obs_n = self._convert_obs(next_obs)

        a_prime = self.get_action(next_state)
        q_s_prime = self.q_table[next_obs_n,:]
        a_star = np.argmax(q_s_prime)

        delta = reward + self.discount_rate * self.q_table[next_obs_n, a_star] - self.q_table[obs_n, action]
        self.E[obs_n, action] += 1 # using accumulating traces

        for s in range(self.obs_space):
            for a in range(self.action_space):
                self.q_table[s, a] += self.learning_rate * delta * self.E[s, a]
                if a_star == a_prime:
                    self.E[s, a] = self.discount_rate * self.lam * self.E[s, a]
                else:
                    self.E[s, a] = 0 # this cuts off eligibiliy from exploratory actions


        self.exploration_rate = self.min_exploration_rate + \
                (self.max_exploration_rate - self.min_exploration_rate) * \
                np.exp(-self.exploration_decay_rate * self.step_counter)
        self.step_counter += 1
        return a_prime

    def save_q_table(self):
        file_location = 'Documents/ToyTrackProblem/'
        np.save(file_location + 'agent_q_table.npy', self.q_table)
        
        self.exploration_rate = self.min_exploration_rate

    def load_q_table(self):
        print("Loaded Q table")
        self.q_table = np.load('agent_q_table.npy')

    def reset_agent(self):
        self.E = np.zeros((self.obs_space, self.action_space))

    def print_params(self):
        print("Explore Rate: " + str(self.exploration_rate))

    def print_q(self):
        for i in range(self.obs_space):
            s_msg = "State: %d "%i + str(np.around(self.q_table[i,:], 3))
            v_msg = " --> Value: " + str(np.around(np.max(self.q_table[i, :]),3))
            print(s_msg + v_msg)


# class LearnedModel:
#     def __init__(self, state_n, action_n):
#         self.r_table = np.zeros((state_n, action_n))
#         self.sp_table = np.zeros((state_n, action_n))

#         self.alpha = 0.1

#     def update_table(self, s, a, r, s_p):
#         self.r_table[s, a] += self.alpha * (r - self.r_table[s,a])
#         # self.sp_table[s, a] += self.alpha * (s_p - self.sp_table[s,a])

# I have merged