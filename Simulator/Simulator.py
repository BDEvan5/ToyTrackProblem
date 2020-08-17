import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image


class CarModel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.wheelbase = 0.33
        self.mass = 3.74
        self.len_cg_rear = 0.17
        self.I_z = 0.047
        self.mu = 0.523
        self.height_cg = 0.074
        self.cs_f = 4.718
        self.cs_r = 5.45

        self.max_d_dot = 3.2
        self.max_steer = 0.4
        self.max_a = 7.5
        self.max_decel = -8.5
        self.max_v = 7.5


    def update_kinematic_state(self, a, d_dot, dt=1):
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.theta = self.theta + self.theta_dot * dt

        a = np.clip(a, self.max_decel, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

    def get_car_state(self):
        state = []
        state.append(self.x)
        state.append(self.y)
        state.append(self.theta)
        state.append(self.velocity)
        state.append(self.steering)

        return state



        

class ScanSimulator:
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise

        self.dth = self.fov / (self.number_of_beams -1)
        self.scan_output = np.zeros(number_of_beams)

        self.step_size = 1
        self.n_searches = 30

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, x, y, theta):

        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i
            self.scan_output[i] = self.trace_ray(x, y, scan_theta)

    def trace_ray(self, x, y, theta):
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * j
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations(self.car_x, dx)
            if self._check_location(search_val):
                break             


    def set_map(self, check_fcn):
        self._check_location = check_fcn


    
        



class F110Env:
    def __init__(self):
        self.timestep = 0.01
        
        self.map_path = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.car = CarModel()
        self.scan_sim = ScanSimulator(10)


    def step(self, action):
        acceleration = action[0]
        steer_dot = action[1]

        self.car.update_kinematic_state(acceleration, steer_dot)
        self.check_done()
        self.update_reward()

        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        return obs, done, reward, _


    def reset(self, poses=None):
        self.scan_sim.set_map(self._check_location)

        self.x = poses['x']
        self.y = poses['y']
        self.theta = poses['theta']
        self.steering = 0
        self.velocity = 0

        

        return self.get_state_obs()


    def get_state_obs(self):
        car_state = self.car.get_car_state()
        scan = self.scan_sim.get_scan(self.car.x, self.car.y, self.car.theta)

        state = np.concatenate([car_state, scan])

        return state

    def init_map(self, map_img_path):
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

    
    def _check_location(self, x):
        if x[0] < 1 or x[0] > self.width:
            return True
        if x[1] < 1 or x[1] > self.height:
            return True 

        if self.race_map[int(round(x[0])), int(round(x[1]))]:
            return True

        return False
