import numpy as np 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls

import random 
from collections import deque
import logging
from matplotlib import pyplot as plt 
from copy import deepcopy
import os


class Model(tf.keras.Model):
    def __init__(self, actions_n):
        super().__init__()

        self.hidden_layer = kl.Dense(128, activation='relu', input_shape=(1,))
        self.output_layer = kl.Dense(actions_n, activation='linear')

    def call(self, inputs, **kwags):
        x = tf.convert_to_tensor(inputs)

        hiddens = self.hidden_layer(x)
        action = self.output_layer(hiddens)

        return action

    
class Agent_ActionValue:
    # basically rewriting the simulation class
    def __init__(self, state_space_n, action_space_n):
        self.gamma = 0.99
        self.value_c = 0.5
        self.entropy_c = 1e-4
        lr = 7e-3
        self.update_network_const = 10
        self.replay_batch_size = 8
        self.epsilon = 0.999
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        # use model to predict and target to update
        self.model = Model(action_space_n)
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            loss='mse')

        self.target = Model(action_space_n)
        self.target.compile(
            optimizer=ko.RMSprop(lr=lr),
            loss='mse')

        # self.prediction_model = PredictionModel()

        self.actions_n = action_space_n
        self.state_n = state_space_n

        self.memory = deque(maxlen=2000)

        self.agent_file_path = "Agent_AV_SimTests/"
        self.agent_test_path = "Agent_A2C_SimTests/AgentTests/"
        self.weight_path = self.agent_file_path + "ModelWeights/target_weights"
        
    def clear_test_files(self):
        file_path_list = ["EpHistories/", "Plots/", "TrainingImages/"]
        for path in file_path_list:
            file_path = self.agent_file_path + path
            for file_name in os.listdir(os.getcwd() + "/" + file_path):
                os.remove(file_path + file_name) # deletes old files
                print("File deleted: " + str(file_name))

    def train(self, env, steps, train_name="TrainName", f_test=20):
        print(train_name)
        self._reset_exploration_rate()
        ep_rewards = [0.0]
        f_weight_update = 5

        next_obs = env.reset()
        for step in range(steps):
            obs = deepcopy(next_obs)
            action = self.get_action(next_obs, env)

            next_obs, reward, done = env.step(action)
            ep_rewards[-1] += reward
            
            self.memorize(obs, action, reward, next_obs, done)

            if done:
                ep_rewards.append(0.0) # this is the value to += for the next episode
                plot(ep_rewards, 10, train_name)

                if len(ep_rewards) % f_test == 1: # test and save every few 
                    self.test(env, train_name + str(len(ep_rewards)))

                if len(ep_rewards) % f_weight_update == 1:
                    self.update_model_weights()

                next_obs = env.reset() # must happen after the training. 

            if len(self.memory) > self.replay_batch_size:
                self.replay()

        self.test(env, train_name + ":FinalTest")
        self.update_model_weights()
        plt.figure(2)
        plt.savefig(self.agent_file_path + "Plots/" + train_name)
        return ep_rewards

    def test(self, env, test_name="TestName"):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action = self.get_action(obs, env)
            obs, reward, done = env.step(action)
            ep_reward += reward
        env.render_episode(self.agent_test_path + test_name)
        print("AV Agent tested, ep reward = %d --> "%ep_reward + test_name)
        env.sim_mem.save_ep(self.agent_test_path + test_name)
        return ep_reward

    def get_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.actions_n-1)
        action_values = self.model.predict(state)[0]
        action = np.argmax(action_values)
        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # print("Replay function: target, target_f")
        minibatch = random.sample(self.memory, self.replay_batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target_prediction = self.model.predict(next_state)
                target = (reward + self.gamma *
                          np.amax(target_prediction[0]))
                # print(target)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.target.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_model_weights(self):
        self.target.save_weights(self.weight_path)
        self.model.load_weights(self.weight_path)

    def load_weights(self):
        self.model.load_weights(self.agent_file_path + "ModelWeights/target_weights")

    def _reset_exploration_rate(self):
        self.epsilon = 0.999
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01


def plot(values, moving_avg_period, title):
    plt.figure(2)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    moving_avg = get_moving_average(moving_avg_period * 2, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    # print("Episode", (len(values)), "\n", \
    #     moving_avg_period, "episode moving avg:", moving_avg)

def get_moving_average(period, values):
    moving_avg = np.zeros_like(values)

    for i, avg in enumerate(moving_avg):
        if i > period:
            moving_avg[i] = np.mean(values[i-period:i])
        # else already zero
    return moving_avg

