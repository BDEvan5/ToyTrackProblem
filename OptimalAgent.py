import numpy as np 

from PathFinder import PathFinder, modify_path
import LibFunctions as lib


class OptimalAgent:
    def __init__(self, race_map):
        self.race_map = race_map
        self.wpts = None

        self.path_name = "DataRecords/" + self.race_map.name + "_path.npy" # move to setup call
        self.pind = 1
        self.target = None


    def init_agent(self):
        # self.race_map.show_hm()
        try:
            raise Exception
            self.wpts = np.load(self.path_name)
        except:
            fcn = self.race_map._check_line_path
            path_finder = PathFinder(fcn, self.race_map.start, self.race_map.end)
            path = path_finder.run_search(5)
            self.race_map.show_map(path)
            path = modify_path(path)
            self.wpts = path
            np.save(self.path_name, self.wpts)
            print("Path Generated")

        self.wpts = np.append(self.wpts, self.race_map.end)
        self.wpts = np.reshape(self.wpts, (-1, 2))

        new_pts = []
        for wpt in self.wpts:
            if not self.race_map._check_location(wpt):
                new_pts.append(wpt)
            else:
                pass
        self.wpts = np.asarray(new_pts)    

        self.race_map.show_map(self.wpts)

        self.pind = 1

        return self.wpts
    
    def act(self, obs):
        # v_ref, d_ref = self.get_corridor_references(obs)
        v_ref, d_ref = self.get_target_references(obs)
        a, d_dot = self.control_system(obs, v_ref, d_ref)

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return [a, d_dot]

    def get_corridor_references(self, obs):
        ranges = obs[5:]
        max_range = np.argmax(ranges)
        dth = np.pi / 9
        theta_dot = dth * max_range - np.pi/2

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        v_ref = 6

        return v_ref, delta_ref

    def get_target_references(self, obs):
        self._set_target(obs)

        v_ref = 6

        th_target = lib.get_bearing(obs[0:2], self.target)
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
