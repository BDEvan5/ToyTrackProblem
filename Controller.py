import LocationState as ls
import numpy as np
import LibFunctions as f
import EpisodeMem as em
import TrackEnv1


class Controller:
    def __init__(self, env, logger):
        self.env = env # at the moment the points are stored in the track - consider chanings
        self.logger = logger

        self.dt = 0.5 # controller frequency

        self.car = TrackEnv1.CarModel()

        self.state = ls.CarState()
        
        self.cur_target = ls.WayPoint()


    def run_control(self, max_steps=100):
        i = 0
        self.state = self.env.reset()
        for w_pt in self.env.track.route: # steps through way points
            self.cur_target = w_pt
            # w_pt.print_point()
            i = 0
            while not self.at_target() and i < max_steps: #keeps going till at each pt
                action = self.get_controlled_action(w_pt)
                self.env.control_step(action)

                self.logger.debug("Current Target: " + str(w_pt.x) + "V_th" + str(w_pt.v) + "@" + str(w_pt.theta))

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
    
    def get_controlled_action(self, wp):
        x_ref = wp.x 
        v_ref = wp.v 
        th_ref = wp.theta

        # run v control
        e_v = v_ref - self.state.v # error for controler
        a = self.acc_control(e_v)

        k_th_ref = 0.1 # amount to favour v direction
        # run th control
        x_ref_th = self.get_xref_th(self.state.x, x_ref)
        e_th = th_ref * k_th_ref + x_ref_th * (1- k_th_ref) # - self.state.theta
        th = self.th_controll(e_th)

        self.logger.debug("X_ref_th: " + str(x_ref_th))

        action = [a, th]
        # print(action)
        return action

    def acc_control(self, e_v):
        # this function is the actual controller
        k = 0.15
        return k * e_v

    def th_controll(self, e_th):
        # theta controller to come here when dth!= th
        return e_th

    def get_xref_th(self, x1, x2):
        dx = x2[0] - x1[0]
        dy = x2[1] - x1[1]
        # self.logger.debug("x1: " + str(x1) + " x2: " + str(x2))
        # self.logger.debug("dxdy: %d, %d" %(dx,dy))
        if dy != 0:
            ret = np.abs(np.arctan(dx / dy))
        else:
            ret = np.pi / 2

        # sort out the sin
        sign = 1
        if dx < 0:
            sign = -1
        if dy > 0: # dy is opposite to normal
            ret = np.pi - np.abs(ret)
        return ret * sign
            




