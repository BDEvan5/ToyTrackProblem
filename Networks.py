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


class Network_AV(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.hidden_layer = kl.Dense(128, activation='relu', input_shape=(1,))
        self.output_layer = kl.Dense(self.config.action_space, activation='linear')

    def call(self, inputs, **kwags):
        x = tf.convert_to_tensor(inputs)

        hiddens = self.hidden_layer(x)
        action = self.output_layer(hiddens)

        return action


    def get_action(self, state):
        action_values = self.predict(state)[0]
        action = np.argmax(action_values)
        return action

    def compile_self(self):
        self.compile(
            optimizer=ko.RMSprop(lr=self.config.lr),
            loss='mse')
