import numpy as np 
from matplotlib import pyplot as plt

import LibFunctions as lib
from RaceTrackMap import TrackMap


class CarModel:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0
        self.th_dot = 0

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
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, a, d_dot, dt):
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.th_dot = theta_dot
        dth = theta_dot * dt
        self.theta = lib.add_angles_complex(self.theta, dth)

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

        self.step_size = 0.1
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
        # obs_res = 10
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * (j + 1)  # search from 1 step away from the point
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        ray = (j) / self.n_searches #* (1 + np.random.normal(0, self.std_noise))
        return ray

    def set_map(self, check_fcn):
        self._check_location = check_fcn


class TrackSim:
    def __init__(self, env_map):
        self.timestep = 0.01

        self.env_map = env_map

        self.car = CarModel()
        self.scan_sim = ScanSimulator(10, np.pi*2/3)
        self.scan_sim.set_map(self.env_map.check_scan_location)

        self.done = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.obs_space = len(self.get_state_obs())
        self.ds = 10

        self.steer_history = []
        self.velocity_history = []
        self.done_reason = ""
        self.y_forces = []

    def step(self, action, updates=10):
        self.steps += 1
        self.action = action
        acceleration = action[0]
        steer_dot = action[1]

        self.car.prev_loc = [self.car.x, self.car.y]
        for _ in range(updates):
            self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
         
        self.check_done_reward_track_train()
        self.steer_history.append(steer_dot)
        self.velocity_history.append(self.car.velocity)

        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        self.action_memory.append([self.car.x, self.car.y])

        return obs, reward, done, None

    def step_cs(self, action):
        self.steps += 1

        v_ref = action[0]
        d_ref = action[1]
        self.action = action

        frequency_ratio = 10 # cs updates per planning update
        self.car.prev_loc = [self.car.x, self.car.y]
        for i in range(frequency_ratio):
            acceleration, steer_dot = self.control_system(v_ref, d_ref)
            self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)

            self.steer_history.append(steer_dot)
            self.velocity_history.append(self.car.velocity)
         
        self.check_done_reward_track_train()

        obs = self.get_state_obs()
        done = self.done
        reward = self.reward

        self.action_memory.append([self.car.x, self.car.y])

        return obs, reward, done, None

    def control_system(self, v_ref, d_ref):

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def reset(self, poses=None, random_start=False):
        self.done = False
        self.action_memory = []
        self.steps = 0
        
        self.car.x = self.env_map.start[0]
        self.car.y = self.env_map.start[1]
        self.car.prev_loc = [self.car.x, self.car.y]
        self.car.velocity = 0
        self.car.steering = 0
        self.car.theta = 0

        return self.get_state_obs()

    def show_history(self):
        plt.figure(3)
        plt.title("Steer history")
        plt.plot(self.steer_history)
        plt.pause(0.001)
        plt.figure(2)
        plt.title("Velocity history")
        plt.plot(self.velocity_history)
        plt.pause(0.001)
        self.velocity_history.clear()
        plt.figure(1)
        plt.title("Forces history")
        plt.plot(self.y_forces)
        plt.pause(0.001)
        self.y_forces.clear()

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

    def check_done_reward_track_train(self):
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        self.y_forces.append(horizontal_force)
        if horizontal_force > self.car.max_friction_force:
            self.done = True
            self.reward = -1
            self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
        if self.steps > 2000:
            self.done = True
            self.done_reason = f"Max steps"
        start_y = self.env_map.start[1]
        if self.car.prev_loc[1] < start_y - 0.5 and self.car.y > start_y - 0.5:
            if abs(self.car.x - self.env_map.start[0]) < 10:
                self.done = True
                self.done_reason = f"Lap complete"

    def render(self, wait=False, wpts=None):
        fig = plt.figure(4)
        plt.clf()  

        c_line = self.env_map.track_pts
        track = self.env_map.track
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0]*self.ds, l_line[:, 1]*self.ds, linewidth=1)
        plt.plot(r_line[:, 0]*self.ds, r_line[:, 1]*self.ds, linewidth=1)

        # plt.imshow(self.env_map.obs_map.T, origin='lower')
        plt.imshow(self.env_map.scan_map.T, origin='lower')

        plt.xlim([0, 100])
        plt.ylim([0, 100])

        plt.plot(self.env_map.start[0]*self.ds, self.env_map.start[1]*self.ds, '*', markersize=12)

        plt.plot(self.env_map.end[0]*self.ds, self.env_map.end[1]*self.ds, '*', markersize=12)
        plt.plot(self.car.x*self.ds, self.car.y*self.ds, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [self.car.x*self.ds, range_val[0]*self.ds]
            y = [self.car.y*self.ds, range_val[1]*self.ds]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0]*self.ds, pos[1]*self.ds, 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            plt.plot(xs, ys)
            # plt.plot(xs, ys, 'x', markersize=20)

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
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(100, 40, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, 35, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
            
    def render_snapshot(self, wait=False, wpts=None):
        fig = plt.figure(4)
        plt.clf()  
        c_line = self.env_map.track_pts
        track = self.env_map.track
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0]*self.ds, l_line[:, 1]*self.ds, linewidth=1)
        plt.plot(r_line[:, 0]*self.ds, r_line[:, 1]*self.ds, linewidth=1)
        ret_map = self.env_map.get_show_map()
        plt.imshow(ret_map.T, origin='lower')

        plt.xlim([0, 100])
        plt.ylim([0, 100])

        plt.plot(self.env_map.start[0]*self.ds, self.env_map.start[1]*self.ds, '*', markersize=12)

        plt.plot(self.env_map.end[0]*self.ds, self.env_map.end[1]*self.ds, '*', markersize=12)
        plt.plot(self.car.x*self.ds, self.car.y*self.ds, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            x = [self.car.x*self.ds, range_val[0]*self.ds]
            y = [self.car.y*self.ds, range_val[1]*self.ds]
            plt.plot(x, y)

        for pos in self.action_memory:
            plt.plot(pos[0]*self.ds, pos[1]*self.ds, 'x', markersize=6)

        if wpts is not None:
            xs, ys = [], []
            for pt in wpts:
                xs.append(pt[0]*self.ds)
                ys.append(pt[1]*self.ds)
        
            plt.plot(xs, ys)
            # plt.plot(xs, ys, 'x', markersize=20)

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
        s = f"Done reason: {self.done_reason}"
        plt.text(100, 40, s) 
        

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

def CorridorCS(obs):
    ranges = obs[5:]
    max_range = np.argmax(ranges)
    dth = (np.pi * 2/ 3) / 9
    theta_dot = dth * max_range - np.pi/3

    ld = 0.3 # lookahead distance
    delta_ref = np.arctan(2*0.33*np.sin(theta_dot)/ld)

    v_ref = 2

    return [v_ref, delta_ref]



def sim_driver():
    race_map = TrackMap()
    env = TrackSim(race_map)

    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = CorridorCS(state)
        s_p, r, done, _ = env.step_cs(action)
        score += r
        state = s_p

        # env.render(True)
        env.render(False)

    print(f"Score: {score}")
    env.show_history()
    env.render_snapshot(True)




if __name__ == "__main__":
    sim_driver()
