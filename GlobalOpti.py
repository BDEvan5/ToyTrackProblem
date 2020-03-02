import numpy as np 
import TrackEnv1 
import LibFunctions as f
import LocationState as ls


class A_Star:
    def __init__(self, track, logger, ds=10):
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
            return True
        return False

    def _check_closed_list(self, new_node):
        for closed_child in self.closed_list:
            if new_node.position == closed_child.position:
                self.logger.debug(closed_child.position)
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

            if self._check_collision(new_position): 
                continue # collision - skp this direction
            # Create new node - no obstacle
            new_node = Node(self.current_node, new_position)

            self.children.append(new_node)
        
        for new_node in self.children:
            if self._check_closed_list(new_node): # no in closed list
                # self.logger.debug("Didn't add CLOSED node")
                # self.logger.debug(new_node.log_msg())
                continue
           
            # Create the f, g, and h values
            new_node.g = self.current_node.g + self.ds
            # child.g = f.get_distance(self.start_n.position, child.position)
            h_val = f.get_distance(new_node.position, self.end_n.position)
            new_node.h = h_val 
            new_node.f = new_node.g + new_node.h

             # Child is already in the open list
            if self._check_open_list(new_node):
                self.logger.debug("Didn't add OPEN node")
                self.logger.debug(new_node.log_msg())
                continue

            # Add the child to the open list

            self.open_list.append(new_node)
            self.open_node_n += 1
            self.logger.debug("Added new node to open list")
            self.logger.debug(new_node.log_msg())
                
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

    def get_opti_path(self):
        ret_list = []
        way_point = [0, 0]
        curr = self.current_node
        while curr is not None:
            ret_list.append(curr.position)
            curr = curr.parent

        closed_list = []
        for node in self.closed_list:
            closed_list.append(node.position)

        return ret_list[::-1], closed_list




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





