import numpy as np 
from matplotlib import pyplot as plt


class CarState:
    def __init__(self):
        self.x = 0
        self. y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.wheelbase = 0.33

    def update_kinematic_state(self, a, d_dot, dt=1):
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.theta = self.theta + self.theta_dot * dt

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt