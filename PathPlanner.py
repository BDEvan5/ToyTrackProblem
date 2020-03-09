import LibFunctions as f 
import numpy as np 



class PathPlanner:
    def __init__(self, track, car, logger):
        self.track = track
        self.logger = logger
        self.car = car

        self.path_finder = A_StarPathFinder(track, logger, 10)

    def plan_path(self):
        self.path_finder.run_search()
        self.add_velocity()

    def add_velocity(self):
        # set up last wp in each cycle
        path = self.track.route
        for i, wp in enumerate(path):
            if i == 0:
                last_wp = wp
                continue
            dx = wp.x[0] - last_wp.x[0]
            dy = wp.x[1] - last_wp.x[1]
            if dy != 0:
                gradient = dx/dy  #flips to make forward theta = 0
            else:
                gradient = 1000
            last_wp.theta = np.arctan(gradient)  # gradient to next point
            last_wp.v = self.car.max_v * (np.pi - last_wp.theta) / np.pi

            last_wp = wp

        path[len(path)-1].theta = 0 # set the last point
        path[len(path)-1].v = self.car.max_v


class A_StarPathFinder:
    def __init__(self, track, logger, ds):
        # ds is the search size around the current node
        self.ds = ds
        self.track = track
        self.logger = logger

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        for i in range(3):
            for j in range(3):
                direction = [(j-1)*self.ds, (i-1)*self.ds]
                self.position_list.append(direction)

        self.position_list.pop(4) # remove stand still
        self.current_node = Node()

        self.open_node_n = 0

    def run_search(self, max_steps=4000):
        self.set_up_start_node()
        i = 0
        while len(self.open_list) > 0 and i < max_steps:
            self.find_current_node()

            if self.check_done():
                print("The shortest path has been found")
                break

            self.generate_children()
            i += 1
        self.set_track_path()

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
        # self.logger.debug(self.current_node.log_msg())

    def check_done(self):
        dx_max = self.ds
        dis = f.get_distance(self.current_node.position, self.end_n.position)
        if dis < dx_max:
            print("Found")
            return True
        return False

    def _check_closed_list(self, new_node):
        for closed_child in self.closed_list:
            if new_node.position == closed_child.position:
                return True
        return False

    def _check_open_list(self, new_node):
        for open_node in self.open_list:
            if new_node == open_node: # if in open set return true
                if new_node.g < open_node.g: # if also beats g score - update g
                    open_node.g = new_node.g
                    open_node.parent = new_node.parent
                return True
        return False

    def generate_children(self):
        self.children.clear() # deletes old children

        for direction in self.position_list:
            new_position = f.add_locations(self.current_node.position, direction)

            if self.track._check_collision(new_position): 
                continue # collision - skp this direction
            # Create new node - no obstacle
            new_node = Node(self.current_node, new_position)

            if self._check_closed_list(new_node): # no in closed list
                # self.logger.debug("Didn't add CLOSED node")
                # self.logger.debug(new_node.log_msg())
                continue
           
            # Create the f, g, and h values
            new_node.g = self.current_node.g + self.ds
            h_val = f.get_distance(new_node.position, self.end_n.position)
            new_node.h = h_val 
            new_node.f = new_node.g + new_node.h

             # Child is already in the open list
            if self._check_open_list(new_node):
                # self.logger.debug("Didn't add OPEN node")
                # self.logger.debug(new_node.log_msg())
                continue

            # Add the child to the open list

            self.open_list.append(new_node)
            self.open_node_n += 1
            # self.logger.debug("Added new node to open list")
            # self.logger.debug(new_node.log_msg())
                
    def set_track_path(self):
        # this sets the path inside the track to the a star path
        curr = self.current_node
        pos_list = []
        while curr is not None:
            pos_list.append(curr.position)
            
            curr = curr.parent
        pos_list = pos_list[::-1]
        for pos in pos_list:
            self.track.add_way_point(pos)

        self.track.print_route()



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

    def log_msg(self):
        if self.parent is not None:
            msg1 = "Parent: " + str(self.parent.position)
            msg2 = "; Pos: " + str(self.position)
            msg3 = "; G: %d -> h: %d -> f: %d " %(self.g, self.h, self.f)

            msg = msg1 + msg2 + msg3
            return msg


class TrackSmoothing:
    # this class modifies the track route so that it is driveable
    def __init__(self, track, car):
        self.car = car
        self.track = track

        self.path_in = None

    def add_path(self, planned_path):
        self.path_in = planned_path

    def add_velocity(self):
        # set up last wp in each cycle
        for i, wp in enumerate(self.track.route.route):
            if i == 0:
                last_wp = wp
                continue
            dx = wp.x[0] - last_wp.x[0]
            dy = wp.x[1] - last_wp.x[1]
            if dy != 0:
                gradient = dx/dy  #flips to make forward theta = 0
            else:
                gradient = 1000
            last_wp.theta = np.arctan(gradient)  # gradient to next point
            last_wp.v = self.car.max_v * (np.pi - last_wp.theta) / np.pi

            last_wp = wp

        self.path_in[len(self.path_in)-1].theta = 0 # set the last point
        self.path_in[len(self.path_in)-1].v = self.car.max_v
    
    def get_path(self):
        return self.path_in