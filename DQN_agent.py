import numpy as np 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kls
import tensorflow.keras.optimizers as klo

import TrackEnv1
import random 
from collections import deque
import logging
from matplotlib import pyplot as plt 

batch_size = 16

class DQN:
    def __init__(self, state_space, action_space):
        self.lr = 0.01
        self.epsilon = 0.99
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.gamma = 0.95

        self.action_size = action_space
        self.memory = deque(maxlen=2000)
        
        self.model = k.Sequential()
        self.model.add(kls.Dense(state_space, activation='relu'))
        self.model.add(kls.Dense(24, activation='relu'))
        self.model.add(kls.Dense(24, activation='relu'))
        self.model.add(kls.Dense(action_space, activation='linear'))
        self.model.compile(loss='mse',
                      optimizer=klo.Adam(lr=self.lr))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])


    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
class AgentDQN:
    # basically rewriting the simulation class
    def __init__(self, track, car):
        logging.basicConfig(filename="Documents/ToyTrackProblem/AgentLog.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.track = track
        self.car = car

        self.env = TrackEnv1.RaceEnv(self.track, self.car, self.logger)
        action_space = 3
        state_space = 32
        self.agent = DQN(state_space, action_space)

        self.env.action_space = action_space

        self.player = None
        self.ep_mem = None

        self.rewards = []

    def run_learning(self, num_episodes):
        self.set_up_track()
        for i in range(num_episodes):
            self.run_episode(i)

        self.env.sim_mem.save_ep("Last_ep")

    def run_episode(self, e):
        max_steps = 200
        ep_reward = 0

        s = self.env.reset()
        for i in range(max_steps):
            obs = s.get_sense_observation() 

            action = self.agent.get_action(obs)

            s_p, r, done = self.env.step(action)
            obs_p = s_p.get_sense_observation()
            print(obs_p)
            ep_reward += r

            self.agent.memorize(obs, action, r, obs_p, done)

            s = s_p
            if done:
                print("episode: {}, score: {}, e: {:.2}"
                    .format(e, ep_reward, self.agent.epsilon))
                break
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)

        self.rewards.append(ep_reward)
        

    def set_up_track(self):
        self.track.add_way_point(self.track.start_location)
        self.track.add_way_point(self.track.end_location)
        self.track.route[1].v = self.car.max_v

    def plot_rewards(self):
        i = range(len(self.rewards))
        # print(self.rewards)
        plt.plot(i, self.rewards, 'x')
        plt.show()

    
