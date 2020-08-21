import numpy as np 

from PathFinder import PathFinder, modify_path
import LibFunctions as lib


class PureRepDataGen:
    def __init__(self, env_map):
        self.env_map = env_map
        
        self.wpts = None
        self.pind = 1
        self.target = None

        self.nn_wpts = None
        self.nn_pind = 1
        self.nn_target = None

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call

    def init_agent(self):
        fcn = self.env_map.obs_hm._check_line
        path_finder = PathFinder(fcn, self.env_map.start, self.env_map.end)
        try:
            path = path_finder.run_search(5)
        except AssertionError:
            print(f"Search Problem: generating new start")
            self.env_map.generate_random_start()
            path = path_finder.run_search(5)
        self.wpts = modify_path(path)
        print("Path Generated")

        self.wpts = np.append(self.wpts, self.env_map.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self.env_map.race_course._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        self.env_map.race_course.show_map(False, self.wpts)

        self.pind = 1

        self.init_straight_plan()        

        return self.wpts

    def init_straight_plan(self):
        # this is when there are no known obs for training.
        start = self.env_map.start
        end = self.env_map.end

        resolution = 10
        dx, dy = lib.sub_locations(end, start)

        n_pts = max((round(max((abs(dx), abs(dy))) / resolution), 3))
        ddx = dx / (n_pts - 1)
        ddy = dy / (n_pts - 1)

        self.nn_wpts = []
        for i in range(n_pts):
            pt = lib.add_locations(start, [ddx, ddy], i)
            self.nn_wpts.append(pt)

        self.nn_pind = 1

    def act(self, obs):
        # v_ref, d_ref = self.get_corridor_references(obs)
        self._set_target(obs)
        v_ref, d_ref = self.get_target_references(obs, self.target)
        a, d_dot = self.control_system(obs, v_ref, d_ref)

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return [a, d_dot]

    def get_nn_vals(self, obs):
        v_ref, d_ref = self.get_target_references(obs, self.nn_target)

        max_angle = np.pi/2
        max_v = 7.5

        target_theta = (lib.get_bearing(obs[0:2], self.nn_target) - obs[2]) / (2*max_angle)
        nn_obs = [target_theta, obs[3]/max_v, obs[4]/max_angle, v_ref/max_v, d_ref/max_angle]
        nn_obs = np.array(nn_obs)

        nn_obs = np.concatenate([nn_obs, obs[5:]])

        return nn_obs

    def get_corridor_references(self, obs):
        ranges = obs[5:]
        max_range = np.argmax(ranges)
        dth = np.pi / 9
        theta_dot = dth * max_range - np.pi/2

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        v_ref = 6

        return v_ref, delta_ref

    def get_target_references(self, obs, target):
        v_ref = 6

        th_target = lib.get_bearing(obs[0:2], target)
        theta_dot = th_target - obs[2]
        theta_dot = lib.limit_theta(theta_dot)

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        return v_ref, delta_ref

    def control_system(self, obs, v_ref, d_ref):
        kp_a = 10
        a = (v_ref - obs[3]) * kp_a
        
        kp_delta = 5
        d_dot = (d_ref - obs[4]) * kp_delta

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance and self.pind < len(self.wpts)-2: # how close to say you were there
            self.pind += 1
        
        self.target = self.wpts[self.pind]

        dis_cur_target = lib.get_distance(self.nn_wpts[self.nn_pind], obs[0:2])
        shift_distance = 5
        if dis_cur_target < shift_distance and self.nn_pind < len(self.nn_wpts)-2: # how close to say you were there
            self.nn_pind += 1
        
        self.nn_target = self.nn_wpts[self.nn_pind]



