import LocationState as ls
import numpy as np
import LibFunctions as f
import EpisodeMem as em
import TrackEnv1


class Controller:
    def __init__(self, env, logger):
        self.env = env # at the moment the points are stored in the track - consider chanings
        self.logger = logger

        self.car = TrackEnv1.CarModel()

        self.state = ls.CarState()
        
        self.cur_target = ls.WayPoint()


    def run_control(self, max_steps=200):
        i = 0
        self.state = self.env.reset()
        for w_pt in self.env.track.point_list: # steps through way points
            self.cur_target = w_pt
            while not self.at_target() and i< max_steps: #keeps going till at each pt
                
                action = self.get_action()
                self.logger.debug("Current Target: " + str(w_pt.x))

                self.state, done = self.env.control_step(action)

                if done:
                    print("Episode Complete")
                    break

                self.logger.debug("")
                i+=1 

    def at_target(self):
        way_point_dis = f.get_distance(self.state.x, self.cur_target.x)
        # print(way_point_dis)
        if way_point_dis < 2:
            print("Way point reached: " + str(self.cur_target.x))
            return True
        return False


    def get_action(self):
        target = self.cur_target.x
        cur = self.state.x
        direc = f.sub_locations(cur, target)

        scale = -0.1
        action = [0, 0]
        for i in range(2):
            action[i] = direc[i] * scale

        # print(action)
        return action




