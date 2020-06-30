import LibFunctions as f 
import numpy as np 
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import collections  as mc


class PathFinder:
    def __init__(self, o_function, start, end):
        # ds is the search size around the current node
        self.ds = None
        self.obstacle_function = o_function

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        self.current_node = Node()
        self.open_node_n = 0

        self.start = start
        self.end = end

    def set_directions(self):
        for i in range(3):
            for j in range(3):
                direction = [(j-1)*self.ds, (i-1)*self.ds]
                self.position_list.append(direction)

        self.position_list.pop(4) # remove stand still

        # this makes it not go diagonal
        # self.position_list = [[1, 0], [0, -1], [-1, 0], [0, 1]]
        for pos in self.position_list:
            pos[0] = pos[0] * self.ds
            pos[1] = pos[1] * self.ds
        # print(self.position_list)

    def run_search(self, ds, max_steps=4000):
        self.ds = ds
        self.set_directions()
        self.set_up_start_node()
        i = 0
        while len(self.open_list) > 0 and i < max_steps:
            self.find_current_node()

            if self.check_done():
                print("The shortest path has been found")
                break
            self.generate_children()
            i += 1

        assert i < max_steps, "Max Iterations reached: problem with search"
        assert len(self.open_list) > 0, "Search broke: no open nodes"
        path = self.get_path_list()

        return path

    def set_up_start_node(self):
        self.start_n = Node(None, self.start)
        self.end_n = Node(None, self.end)

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
        dx_max = self.ds * 3
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

            if self.obstacle_function(new_position): 
                continue 
            new_node = Node(self.current_node, new_position)

            if self._check_closed_list(new_node): 
                continue
           
            # Create the f, g, and h values
            new_node.g = self.current_node.g + self.ds
            h_val = f.get_distance(new_node.position, self.end_n.position)
            new_node.h = h_val 
            new_node.f = new_node.g + new_node.h

             # Child is already in the open list
            if self._check_open_list(new_node):
                continue

            self.open_list.append(new_node)
            self.open_node_n += 1

    def get_path_list(self):
        curr = self.current_node
        pos_list = []
        while curr is not None:
            pos_list.append(curr.position)
            curr = curr.parent
        pos_list = pos_list[::-1]

        pos_list.append(self.end)

        return pos_list


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