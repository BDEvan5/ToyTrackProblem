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

        self.step_size = 0.2
        self.n_searches = 20

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

    def set_check_fcn(self, check_fcn):
        self._check_location = check_fcn


class TrackSim:
    def __init__(self, env_map):
        self.timestep = 0.02

        self.env_map = env_map

        self.car = CarModel()

        self.done = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.steer_history = []
        self.velocity_history = []
        self.done_reason = ""
        self.y_forces = []

    def step(self, action):
        self.steps += 1

        v_ref = action[0]
        d_ref = action[1]
        self.action = action

        frequency_ratio = 10 # cs updates per planning update
        self.car.prev_loc = [self.car.x, self.car.y]
        for i in range(frequency_ratio):
            acceleration, steer_dot = self.control_system(v_ref, d_ref)
            self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)

        #     self.steer_history.append(steer_dot)
        #     self.velocity_history.append(self.car.velocity)
        self.steer_history.append(self.car.steering)
        self.velocity_history.append(self.car.velocity)


        
        self.check_done_reward_track_train()
        # self.check_done_forest()

        obs = self.car.get_car_state()
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
        # self.car.theta = 0
        self.car.theta = np.pi/2

        return self.car.get_car_state()

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
        # plt.figure(1)
        # plt.title("Forces history")
        # plt.plot(self.y_forces)
        # plt.pause(0.001)
        self.y_forces.clear()

    def reset_lap(self):
        self.steps = 0
        self.reward = 0
        self.car.prev_loc = [self.car.x, self.car.y]
        self.action_memory.clear()
        self.done = False

    def check_done_reward_track_train(self):
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        self.y_forces.append(horizontal_force)
        if horizontal_force > self.car.max_friction_force:
            # self.done = True
            self.reward = -1
            self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
        if self.steps > 500:
            self.done = True
            self.done_reason = f"Max steps"

        car = [self.car.x, self.car.y]
        if lib.get_distance(car, self.env_map.start) < 2 and self.steps > 50:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"

    def check_done_forest(self):
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        self.y_forces.append(horizontal_force)
        if horizontal_force > self.car.max_friction_force:
            # self.done = True
            self.reward = -1
            self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
        if self.steps > 100:
            self.done = True
            self.done_reason = f"Max steps"
        if abs(self.car.theta) > 0.95*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1

        car = [self.car.x, self.car.y]
        if lib.get_distance(car, self.env_map.end) < 2 and self.steps > 10:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"


    def render(self, wait=False, scan_sim=None):
        self.env_map.render_map(4)
        fig = plt.figure(4)

        if scan_sim is not None:
            for i in range(scan_sim.number_of_beams):
                angle = i * scan_sim.dth + self.car.theta - scan_sim.fov/2
                fs = scan_sim.scan_output[i] * scan_sim.n_searches * scan_sim.step_size
                dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
                range_val = lib.add_locations([self.car.x, self.car.y], dx)
                cx, cy = self.env_map.convert_position([self.car.x, self.car.y])
                rx, ry = self.env_map.convert_position(range_val)
                x = [cx, rx]
                y = [cy, ry]
                plt.plot(x, y)

        for pos in self.action_memory:
            x, y = self.env_map.convert_position(pos)
            plt.plot(x, y, 'x', markersize=6)

        text_x = self.env_map.scan_map.shape[1] + 10
        text_y = self.env_map.scan_map.shape[0] / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def min_render(self, wait=False):
        fig = plt.figure(4)
        plt.clf()  

        ret_map = self.env_map.scan_map
        plt.imshow(ret_map)

        # plt.xlim([0, self.env_map.width])
        # plt.ylim([0, self.env_map.height])

        s_x, s_y = self.env_map.convert_to_plot(self.env_map.start)
        plt.plot(s_x, s_y, '*', markersize=12)

        c_x, c_y = self.env_map.convert_to_plot([self.car.x, self.car.y])
        plt.plot(c_x, c_y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            r_x, r_y = self.env_map.convert_to_plot(range_val)
            x = [c_x, r_x]
            y = [c_y, r_y]

            plt.plot(x, y)

        for pos in self.action_memory:
            p_x, p_y = self.env_map.convert_to_plot(pos)
            plt.plot(p_x, p_y, 'x', markersize=6)

        text_start = self.env_map.width + 10
        spacing = int(self.env_map.height / 10)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_start, spacing*1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_start, spacing*2, s) 
        s = f"Done: {self.done}"
        plt.text(text_start, spacing*3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_start, spacing*4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_start, spacing*5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_start, spacing*6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_start, spacing*7, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(text_start, spacing*8, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, spacing*9, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
            

def CorridorCS(obs):
    ranges = obs[5:]
    max_range = np.argmax(ranges)

    wa = 0
    for i in range(10):
        wa += ranges[i] * i
    w_range = wa / 9

    max_range = int(round(w_range))

    dth = (np.pi * 2/ 3) / 9
    theta_dot = dth * max_range - np.pi/3

    ld = 0.5 # lookahead distance
    delta_ref = np.arctan(2*0.33*np.sin(theta_dot)/ld)
    delta_ref = np.clip(delta_ref, -0.4, 0.4)

    v_ref = 2

    return [v_ref, delta_ref]



def sim_driver():
    # race_map = TrackMap()
    race_map = MinMapNpy('torino')
    env = TrackSim(race_map)

    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = CorridorCS(state)
        s_p, r, done, _ = env.step(action)
        score += r
        state = s_p

        # env.min_render(True)
        env.min_render(False)

    print(f"Score: {score}")
    env.show_history()
    env.min_render(True)
    # env.render_snapshot(True)




if __name__ == "__main__":
    sim_driver()
