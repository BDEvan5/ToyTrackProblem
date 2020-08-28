import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib


class CarModel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0

        self.prev_loc = 0

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

    def update_kinematic_state(self, a, d_dot, dt):
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.theta = self.theta + theta_dot * dt

        a = np.clip(a, self.max_decel, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

        self.theta = lib.limit_theta(self.theta)

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
        self.n_searches = 50

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, x, y, theta):
        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i - self.fov/2
            self.scan_output[i] = self.trace_ray(x, y, scan_theta)

        return self.scan_output

    def trace_ray(self, x, y, theta, noise=True):
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * j
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        ray = j / self.n_searches * (1 + np.random.normal(0, self.std_noise))
        return ray

    def set_map(self, check_fcn):
        self._check_location = check_fcn


class F110Env:
    def __init__(self, env_map):
        self.timestep = 0.01

        self.env_map = env_map
        self.race_map = env_map.race_course

        self.car = CarModel()
        self.scan_sim = ScanSimulator(10, np.pi*2/3)
        self.scan_sim.set_map(self.race_map._check_location)

        self.done = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.obs_space = len(self.get_state_obs())

    def step(self, action, updates=10, race=False):
        self.steps += 1
        self.action = action
        acceleration = action[0]
        steer_dot = action[1]

        self.car.prev_loc = [self.car.x, self.car.y]
        for _ in range(updates):
            self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
        
        if race:
            self.reward += updates * self.timestep
            self.check_done_race()
        else:
            self.check_done_reward()      


        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        self.action_memory.append([self.car.x, self.car.y])

        return obs, reward, done, None

    def reset(self, poses=None, random_start=False):
        self.done = False
        self.action_memory = []
        self.steps = 0
        
        if poses is not None:
            self.car.x = poses['x']
            self.car.y = poses['y']
            self.car.theta = poses['theta']
            self.car.steering = 0
            self.car.velocity = 0
        else:
            self.car.x = self.env_map.start[0]
            self.car.y = self.env_map.start[1]
            self.car.velocity = 0
            self.car.steering = 0
            th = lib.get_bearing(self.env_map.start, self.env_map.end) 
            self.car.theta = th + np.random.random() - 0.5
            # self.car.theta = 0

        
        return self.get_state_obs()

    def reset_lap(self):
        self.steps = 0
        self.reward = 0
        self.car.prev_loc = [self.car.x, self.car.y]
        self.action_memory.clear()
        self.done = False

    def get_state_obs(self):
        car_state = self.car.get_car_state()
        scan = self.scan_sim.get_scan(self.car.x, self.car.y, self.car.theta)

        state = np.concatenate([car_state, scan])

        return state

    def check_done_reward(self):
        self.reward = 0 # normal
        if self.race_map._check_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
        if lib.get_distance([self.car.x, self.car.y], self.env_map.end) < 10:
            self.done = True
            self.reward = 1
        if self.steps > 100:
            self.done = True

    def check_done_race(self):
        if self.race_map._check_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
        if self.steps > 400:
            self.done = True
        start_y = self.env_map.start_y
        # counter clock wise
        if self.car.prev_loc[1] < start_y and self.car.y > start_y:
            if abs(self.car.x - self.env_map.start[0]) < 10:
                self.done = True
        # clockwise
        # if self.car.prev_loc[1] < start_y and self.car.y > start_y:
        #     if abs(self.car.x - self.env_map.start[0]) < 10:
        #         self.done = True
    
    def render(self, wait=False, wpts=None):
        car_x = int(self.car.x)
        car_y = int(self.car.y)
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.race_map.T, origin='lower')
        plt.xlim(0, self.race_map.map_width)
        plt.ylim(-10, self.race_map.map_height)
        plt.plot(self.env_map.start[0], self.env_map.start[1], '*', markersize=12)

        plt.plot(self.env_map.end[0], self.env_map.end[1], '*', markersize=12)
        plt.plot(self.car.x, self.car.y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0], pos[1], 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0])
                ys.append(pt[1])
        
            # plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=20)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(100, 45, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
            
    def render_snapshot(self, wait=False, wpts=None):
        self.race_map = self.env_map.race_course

        car_x = int(self.car.x)
        car_y = int(self.car.y)
        fig = plt.figure(4)
        plt.clf()  
        plt.imshow(self.race_map.race_map.T, origin='lower')
        plt.xlim(0, self.race_map.map_width)
        plt.ylim(-10, self.race_map.map_height)
        plt.plot(self.env_map.start[0], self.env_map.start[1], '*', markersize=12)

        plt.plot(self.env_map.end[0], self.env_map.end[1], '*', markersize=12)
        plt.plot(self.car.x, self.car.y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [car_x, range_val[0]]
            y = [car_y, range_val[1]]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0], pos[1], 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0])
                ys.append(pt[1])
        
            # plt.plot(xs, ys)
            plt.plot(xs, ys, 'x', markersize=20)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(100, 80, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(100, 70, s) 
        s = f"Done: {self.done}"
        plt.text(100, 65, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(100, 60, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(100, 55, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(100, 50, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(100, 45, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()


def CorridorAction(obs):
    # all this does is go in the direction of the biggest range finder
    ranges = obs[5:]
    max_range = np.argmax(ranges)
    dth = np.pi / 9
    theta_dot = dth * max_range - np.pi/2

    kp_delta = 5
    L = 0.33
    # d_dot = d_heading * kp_delta
    delta = np.arctan(theta_dot * L / (obs[3]+0.001))
    d_dot = (delta - obs[4]) * kp_delta

    a = 0
    if obs[3] < 6:
        a = 8

    return [a, d_dot]



def sim_driver():
    race_map = TestMap()
    race_map.map_1000(True)
    env = F110Env(race_map)

    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = CorridorAction(state)
        s_p, r, done, _ = env.step(action, updates=20)
        score += r
        state = s_p

        # env.render(True)
        env.render(False)

    print(f"Score: {score}")

if __name__ == "__main__":
    sim_driver()
