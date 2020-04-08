import numpy as np 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls

import random 
from matplotlib import pyplot as plt 
from copy import deepcopy
import os


# class Network_AV(tf.keras.Model):
#     def __init__(self, config):
#         super().__init__()

#         self.config = config

#         self.hidden_layer = kl.Dense(128, activation='relu', input_shape=(1,))
#         self.output_layer = kl.Dense(self.config.action_space, activation='linear')

#     def call(self, inputs, **kwags):
#         x = tf.convert_to_tensor(inputs)

#         hiddens = self.hidden_layer(x)
#         action = self.output_layer(hiddens)

#         return action

#     def get_action(self, state):
#         action_values = self.predict(state)[0]
#         action = np.argmax(action_values)
#         return action

#     def compile_self(self):
#         self.compile(
#             optimizer=ko.RMSprop(lr=self.config.lr),
#             loss='mse')



class ProbDist(tf.keras.Model):
    # this is a model which takes in log probabilities and returns an action by sampling methods
    def call(self, logits, **kwags):
        print("Logits: " + str(logits))
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class NetworkA2C(tf.keras.Model):
    def __init__(self, config):
        super().__init__('mlp_policy') # not sure what that is
        self.config = config

        self.hidden_values = kl.Dense(128, activation='relu', input_shape=(1,))
        self.values = kl.Dense(1, name='values')

        self.hidden_actor = kl.Dense(128, activation='relu', input_shape=(1,))
        self.logits = kl.Dense(config.action_space, name='policy_logits')

        self.dist = ProbDist()

        self.lossfcn = LossA2C(config)

    def call(self, inputs, **kwags):
        x = tf.convert_to_tensor(inputs)
        # print("Tensor x in call")
        # print(x)

        hidden_logs = self.hidden_actor(x)
        logits = self.logits(hidden_logs)

        hidden_values = self.hidden_values(x)
        values = self.values(hidden_values)

        return logits, values

    def action_value(self, obs):
        # this apparently executes call
        logits, value = self.predict_on_batch(obs)
        # print("ActionValueLogits: " + str(logits))
        action = self.dist.predict_on_batch(logits)

        action_ret = np.squeeze(action, axis=-1)
        value_ret = np.squeeze(value, axis=-1)

        return action_ret, value_ret
    
    def compile_self(self):
        self.compile(
            optimizer=ko.RMSprop(lr=self.config.lr),
            loss=[self.lossfcn.logits_loss, self.lossfcn.value_loss])


class LossA2C:
    def __init__(self, config):
        # this is a class to hold the loss functions as they are separate from the agent
        self.config = config

    def value_loss(self, returns, value):
        loss = self.config.value_c * kls.mean_squared_error(returns, value)

        return loss

    def logits_loss(self, actions_advantages, log_probs):
        actions, advs = tf.split(actions_advantages, 2, axis=-1)

        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, log_probs, sample_weight=advs)

        probs = tf.nn.softmax(log_probs)

        entropy_loss = kls.categorical_crossentropy(probs, probs)

        loss = policy_loss - self.config.entropy_c * entropy_loss

        return loss

