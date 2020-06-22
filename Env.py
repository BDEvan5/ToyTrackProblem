import numpy as np 

import LibFunctions as lib


class MakeEnv:
    def __init__(self, track):
        self.track = track

        self.vehicle_model = SimulationCarState(5, 1)
        self.steps = 0

    def step(self, action):
        self.steps += 1
        x = self.vehicle_model.evaluate_new_position(action)
        if self.track.check_collision(x, True):
            done = True
            reward = -1
            state = self.vehicle_model.get_state_obs()
            print("Collision")
            return state, reward, done
        else:
            self.vehicle_model.update_position(x)
            self._update_ranges()
            reward, done = self._get_training_reward()
            state = self.vehicle_model.get_state_obs()

            return state, reward, done

    def reset(self):
        self.start, self.end = self.track.random_start_end()
        self.vehicle_model.reset_location(self.start, self.end)
        # self.track.reset_obstacles()
        self.steps = 0

        return self.vehicle_model.get_state_obs()

    def _get_training_reward(self):
        if self.steps > 200: #max steps
            print("max steps reached ")
            return 0, True # no reward but it is done
        end_dis = lib.get_distance(self.vehicle_model.x, self.track.path_end_location)
        if end_dis < 10: # done
            print("Target reached ")
            return 1, True
        # else get proportional reward
        done = False
        reward = - end_dis / 100 # this normalises it.

        return reward, done


    def _update_ranges(self):
        dx = 5 # search size
        curr_x = self.vehicle_model.x
        th_car = self.vehicle_model.th
        for ran in self.vehicle_model.ranges:
            th = th_car + ran[0]
            crash_val = 0
            i = 0
            while crash_val == 0:
                addx = [dx * i * np.sin(th), -dx * i * np.cos(th)] # check
                x_search = lib.add_locations(curr_x, addx)
                crash_val = self.track.check_collision(x_search, True)
                i += 1
            update_val = (i - 2) * dx # sets the last distance before collision 
            ran[1] = update_val

    def ss(self):
        return self.vehicle_model.get_replay_obs()


""" This is a class to hold the car data inside the simulation"""
class SimulationCarState:
    def __init__(self, n_ranges=5, dt=1):
        self.dt = dt
        self.x = [0, 0]
        self.th = 0
        self.end = None

        self.n_ranges = n_ranges
        self.ranges = np.zeros((n_ranges, 2)) # holds the angle and value
        dth = np.pi / (n_ranges-1)
        for i in range(n_ranges):
            angle = dth * i - np.pi/2
            self.ranges[i, 0] = angle
            self.ranges[i, 1] = 1 # no boundary

        self.friction = 0.1
        self.L = 1

    def evaluate_new_position(self, action):
        # action range = [-1, 1]
        theta = action * np.pi # changes range to full 2pi

        x = [0.0, 0.0]
        r = 2
        x[0] = r * np.sin(theta) + self.x[0]
        x[1] = - r * np.cos(theta) + self.x[1]
        
        return x

    def update_position(self, x):
        self.th = -np.tan(lib.get_gradient(self.x, x)**-1) + np.pi
        self.x = x

    def get_state_obs(self):
        # max normalisation constants
        max_theta = np.pi
        max_range = 100

        relative_end = lib.sub_locations(self.end, self.x)
        d_end = np.linalg.norm(relative_end)
        th_end = np.tan((relative_end[0]/relative_end[1])) # x/y
        # should i scale these? probably a good idea?
        try:
            th_end = th_end[0]
        except:
            pass

        state = []
        state.append(d_end)
        state.append(th_end)

        for ran in self.ranges:
            r_val = np.around((ran[1]/max_range), 4)
            state.append(r_val)

        state = np.array(state)
        # state = state[None, :]
        return state

    def get_replay_obs(self):
        # max normalisation constants
        max_theta = np.pi
        max_range = 100

        state = []
        state.append(float(self.x[0]))
        state.append(float(self.x[1]))
        state.append(float(self.th))

        for ran in self.ranges:
            r_val = np.around((ran[1]/max_range), 4)
            state.append(r_val)

        state = np.array(state)
        # state = state[None, :]
        return state


    def reset_location(self, start_location, end_location):
        self.x = start_location
        self.end = end_location

