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

from Config import config
from RaceEnv import RaceEnv


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


class AgentMCTS_AV:
    def __init__(self, config):
        self.config = config

        # use model to predict and target to update
        self.model = Model(action_space_n)
        self.model.compile(
            optimizer=ko.RMSprop(lr=lr),
            loss='mse')

        self.target = Model(action_space_n)
        self.target.compile(
            optimizer=ko.RMSprop(lr=lr),
            loss='mse')

        self.agent_file_path = "Agent_AV_SimTests/"
        self.agent_test_path = "Agent_A2C_SimTests/AgentTests/"
        self.weight_path = self.agent_file_path + "ModelWeights/target_weights"

    def run_ep(self, env):
        next_state, done = env.reset(), False
        ep_reward = 0
        while not done:
            state = deepcopy(next_state)
            action = self.get_action(state)

            next_state, reward, done = env.step(action)
            ep_reward += reward

            self.ReplayBuffer.save_step((state, action, reward, next_state, done))

        return ep_reward

    def train_on_batch(self, batch):
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target_prediction = self.model.predict(next_state)
                target = (reward + self.gamma *
                          np.amax(target_prediction[0]))
                # print(target)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.target.fit(state, target_f, epochs=1, verbose=0)

            self.config.step_eps()

    def update_model_weights(self):
        self.target.save_weights(self.weight_path)
        self.model.load_weights(self.weight_path)

    def get_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.actions_n-1)
        action_values = self.model.predict(state)[0]
        root = self.create_node(state)
        action = run_mcts(root)
        # action = np.argmax(action_values)
        return action

    def create_node(self, state, prior=0):
        action_values = self.model.predict(state)[0]
        node = SearchNode(0, state, action_values)
        return node

    def run_mcts(self, root):
        for i in range(config.n_simulations):
            node = root # reset current node to start root
            search_path = [node]  # start search path tree

            while node.exanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            parent.expand_node()

    def expand_node(self, node):
        # generates child nodes
        policy_sum = sum(node.action_values)
        # for action in range(self.config.action_space):
            

        for i, action_val in enumerate(node.action_values):
            child_state = self.env.model_step(state, i)
            node.children[i] = self.create_node(child_state, action_val/policy_sum)

		node.state = network_output.hiddenstate
		node.reward = output.reward
		policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
		policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = SearchNode(p / policy_sum)
		    # prior is how good an action is relative to other actions

    def network_output(self, state, action):
        action_vals = self.model.predict_on_batch(state)[0]
        value = max(action_vals)
        next_state, reward = self.env.model_step(state)


class SearchNode:
    def __init__(self, prior: float, state, action_values):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = state
        self.reward = 0
        self.action_values = action_values

    def exanded(self):
		return len(self.children) > 0

	def value(self):
		if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

	def select_child(self):
		if self.visit_count == 0: #ucb scores zero
			return random.sample(self.children.items(), 1)[0] 
		# selects child with highest ucb score.
		_, action, child = max((self.ucb_score(child), action, child) for
											action, child in node.children.items())
		return action, child

	def ucb_score(self, child):
		pb_c = math.log((self.visit_count + config_const)/const_config)
		pb_c *= math.sqrt(self.visit_count)/(child.visit_count + 1)

		prior_score = pb_c * child.prior
		v_score = normalise(child.value())
		return v_score + prior score


# class EnvModel:
#     def __init__(self):

#     def predict(self, state):
#         v = state[0]
#         th = state[1]
#         ac = state[2]
#         dth = state[3]

#         dt = 1 # from env
#         r = dt * v
#         dx = r * np.sin(dth)
#         dy = r * np.cos(dth)

#         vp = v + ac * dt - 0.1 * v
#         thp = dth
#         # assume action remains const

#         new_state = [vp, thp, ac, dth]

#         dth = np.pi / len(state[3:])
#         for i in range(len(state) - 4):
#             th_r = i * dth - np.pi / 2
#             if th_r < 0.001:
#                 new_state.append(state[i + 3])
#             else:
#                 rp = r + dy / np.cos(th_r)
#                 new_state.append(rp)


	

class ReplayBuffer:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_step(self, step):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample_batch(self):
        if len(self.buffer) > self.batch_size:
            sample_batch = random.sample(self.buffer, self.batch_size)
            return sample_batch
        sample_batch = random.sample(self.buffer, len(self.buffer)-1)
        return sample_batch

