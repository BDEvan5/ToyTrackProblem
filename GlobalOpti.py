import numpy as np 
import TrackEnv1 
import LibFunctions as f
import LocationState as ls

# class GlobalPlanner:
#     def __init__(self, track, n_nodes=100):
#         self.track = track
#         self.n_nodes = n_nodes

#     def find_optimum_route(self):

#     def get_first_guess(self, dx1 = 5):
#         # dx1 is first step size


class A_Star:
    def __init__(self, track, ds=5):
        # ds is the search size around the current node
        self.ds = ds
        self.track = track

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        for i in range(3):
            for j in range(3):
                direction = [(j-1)*self.ds, (i-1)*self.ds]
                self.position_list.append(direction)
        print(self.position_list)

        self.current_node = Node()

    def run_search(self, max_steps=500):
        self.set_up_start_node()

        for i in range(max_steps):
            self.find_current_node()

            if self.check_done():
                break

            self.generate_children()

            self.set_up_children()



    def set_up_start_node(self):
        self.start_n = Node(None, self.track.start_location)
        self.end_n = Node(None, self.track.end_location)

        self.open_list.append(self.start_n)

    def find_current_node(self):
        self.current_node = self.open_list[0]
        current_index = 0
        for index, item in enumerate(self.open_list):
            if item.f < self.current_node.f:
                self.current_node = item
                current_index = index
        # Pop current off open list, add to closed list
        self.open_list.pop(current_index)
        self.closed_list.append(self.current_node)

    def check_done(self, dx_max=5):
        dis = f.get_distance(self.current_node.position, self.end_n.position)
        if dis < dx_max:
            return True
        return False

    def generate_children(self):
        self.children.clear() # deletes old children

        for direction in self.position_list:
            new_position = f.add_locations(self.current_node.position, direction)

            if not self._check_collision(new_position): # no collision
                # Create new node - no obstacle
                new_node = Node(self.current_node, new_position)

                # Create the f, g, and h values
                new_node.g = self.current_node.g + 1
                h_val = f.get_distance(new_node.position, self.end_n.position)
                new_node.h = h_val
                new_node.f = new_node.g + new_node.h

                # Append
                self.children.append(new_node)

    def _check_collision(self, x):
        # consider moving to track object
        b = self.track.boundary
        ret = 0
        for o in self.track.obstacles:
            if o[0] < x[0] < o[2]:
                if o[1] < x[1] < o[3]:
                    msg = "Boundary collision --> x: %d;%d"%(x[0],x[1])
                    ret = 1
        if x[0] < b[0] or x[0] > b[2]:
            msg = "X wall collision --> x: %d, b:%d;%d"%(x[0], b[0], b[2])
            ret = 1
        if x[1] < b[1] or x[1] > b[3]:
            msg = "Y wall collision --> y: %d, b:%d;%d"%(x[1], b[1], b[3])
            ret = 1
        # if ret == 1:
        #     # print(msg)
        #     self.logger.info(msg)
        return ret

    def set_up_children(self):
        for child in self.children:
             # Child is on the closed list
            for closed_child in self.closed_list:
                if child == closed_child: # this uses the __eq__
                    continue

            # Child is already in the open list
            for open_node in self.open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            self.open_list.append(child)

    def get_open_list(self):
        ret_list = []
        way_point = [0, 0]
        for node in self.closed_list:
            way_point = node.position
            ret_list.append(way_point)

        return ret_list



class Node():
    """A node class for A* Pathfinding"""
    # helper class for a star

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position







# class Controller:
#     def __init__(self):
#         self.

#     def get_acc(self, x0, x1, v0, v1):
#         dx = f.sub_locations(x1, x0) 
#         dv = f.sub_locations(v1, v0)
#         dt = 2 * np.divide(dx, dv)
#         a = np.divide(dv, [dt, dt])

#         return a, dt


