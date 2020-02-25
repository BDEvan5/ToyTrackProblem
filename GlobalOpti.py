import numpy as np 
import TrackEnv1 
import LibFunctions as f

class GlobalPlanner:
    def __init__(self, n_nodes=100):
        self.n_nodes = n_nodes

    def find_optimum_route(self):




class Controller:
    def __init__(self):
        self.

    def get_acc(self, x0, x1, v0, v1):
        dx = f.sub_locations(x1, x0) 
        dv = f.sub_locations(v1, v0)
        dt = 2 * np.divide(dx, dv)
        a = np.divide(dv, [dt, dt])

        return a, dt


        
