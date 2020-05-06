import LibFunctions as f 
import numpy as np 
from copy import deepcopy
from Interface import Interface
from Models import TrackData

from collections import deque
import matplotlib.pyplot as plt
from matplotlib import collections  as mc


"""
A* algorithm and helpers - for standard shortest path search
"""
class A_StarPathFinder:
    def __init__(self, track):
        # ds is the search size around the current node
        self.ds = None
        self.track = track

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        
        self.current_node = Node()

        self.open_node_n = 0

    def set_directions(self):
        # for i in range(3):
        #     for j in range(3):
        #         direction = [(j-1)*self.ds, (i-1)*self.ds]
        #         self.position_list.append(direction)

        # self.position_list.pop(4) # remove stand still

        # this makes it not go diagonal
        self.position_list = [[1, 0], [0, -1], [-1, 0], [0, 1]]
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
        path = self.set_track_path()

        return path

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
        path = Path()
        for pos in pos_list:
            path.add_way_point(pos)
        path.add_way_point(self.track.end_location)
        return path


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


"""
A* algorithm and helpers - modified for state space
"""
class A_StarPathFinderModified:
    def __init__(self, track):
        # ds is the search size around the current node
        self.ds = None
        self.track = track

        self.open_list = []
        self.closed_list = []
        self.children = []
        
        self.position_list = []
        
        self.current_node = Node()

        self.open_node_n = 0

    def set_directions(self):
        # for i in range(3):
        #     for j in range(3):
        #         direction = [(j-1)*self.ds, (i-1)*self.ds]
        #         self.position_list.append(direction)

        # self.position_list.pop(4) # remove stand still

        # this makes it not go diagonal
        self.position_list = [[1, 0], [0, -1], [-1, 0], [0, 1]]
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
        path = self.set_track_path()

        return path

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
            corner_cost = 1 # per radian of difference
            new_pos = new_node.position
            cur_pos = self.current_node.position
            if self.current_node.parent:
                old_pos = self.current_node.parent.position
                d_cost = get_track_segment_cost(new_pos, cur_pos, old_pos)
            else:
                d_cost = self.ds
            new_node.g = self.current_node.g + d_cost
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
        path = Path()
        for pos in pos_list:
            path.add_way_point(pos)
        path.add_way_point(self.track.end_location)
        return path

def get_track_segment_cost(new_pt, cur_point, old_point):

    dis = f.get_distance(cur_point, new_pt) 
    angle = f.get_angle(old_point, cur_point, new_pt)

    cost = dis + angle ** 2

    return cost

"""
RRT algorithm and helpers
"""
class RRT_PathFinder:
    def __init__(self, track):
        self.ds = None
        self.track = track
        self.G = None

    def run_search(self, ds=5, lim=2000):
        self.G = Graph(self.track.start_location, self.track.end_location)
        self.ds = ds
        G, track = self.G, self.track

        counter = 0

        while counter < lim:
            counter += 1
            x_new = G.random_postion()
            if track._check_collision(x_new):
                continue
            near_vex, near_idx = self.nearest(x_new)
            if near_idx is None:
                continue # collided

            newidx = G.add_vertex(x_new)
            dis = f.get_distance(x_new, near_vex)
            G.add_edge(newidx, near_idx, dis)

            end_dis = f.get_distance(x_new, self.track.end_location)
            if end_dis < self.ds:
                endidx = G.add_vertex(G.endpos)
                G.add_edge(newidx, endidx, end_dis)
                break
            print(counter)

        print("Success")
        endidx = G.add_vertex(G.endpos)
        G.add_edge(newidx, endidx, end_dis)
        path = self.get_path()
        # plot(G, dijkstra(self.G))
        return path

    def get_path(self):
        path_graph = dijkstra(self.G)
        path = Path()
        for node in path_graph: # check node data type
            path.add_way_point(node)

        return path

    def nearest(self, x_new):
        G, track = self.G, self.track
        minDist = float("inf")
        Nidx = None
        nvex = None
        for idx, v in enumerate(G.vertices):
            if track.check_line_collision(v, x_new):
                continue
            dis = f.get_distance(v, x_new)
            if dis < minDist:
                minDist = dis
                Nidx = idx
                nvex = v

        return nvex, Nidx


class Graph:
    def __init__(self, startpos, endpos):
        self.startpos = tuple(startpos)
        self.endpos = tuple(endpos)

        self.vertices = [self.startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {self.startpos:0}
        self.neighbors = {0:[]}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vertex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx


    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def random_postion(self):
        rx = np.random.random()
        ry = np.random.random()

        posx = rx * 100
        posy = ry * 100

        # posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        # posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return (posx, posy)


def dijkstra(G):
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)


def plot(G, path=None):
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots()

    # for obs in obstacles:
    #     circle = plt.Circle(obs, radius, color='red')
    #     ax.add_artist(circle)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()


"""
Helper for all classes - the path that is returned
"""
class Path:
    def __init__(self):
        self.route = []
    
    def add_way_point(self, x, v=0, theta=0):
        wp = WayPoint()
        wp.set_point(x, v, theta)
        self.route.append(wp)

    def print_route(self):
        for wp in self.route:
            print("X: (%d;%d), v: %d, th: %d" %(wp.x[0], wp.x[1], wp.v, wp.theta))

    def __len__(self):
        return len(self.route)

    def __iter__(self):
        for obj in self.route:
            return obj

class WayPoint:
    def __init__(self):
        self.x = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0.0

    def set_point(self, x, v, theta):
        self.x = x
        self.v = v
        self.theta = theta

    def print_point(self):
        print("X: " + str(self.x) + " -> v: " + str(self.v) + " -> theta: " +str(self.theta))

    def __eq__(self, other):
        dx = self.x[0] - other.x[0]
        dy = self.x[1] - other.x[1]
        # strictly not true but done for smoothing ease
        if dx == 0 or dy == 0:
            return True
        return False


def ReduceRoute(path):
    new_path = Path()
    new_path.add_way_point(path.route[0].x)
    pt1 = path.route[0]
    for i in range(2, len(path)):
        pt2 = path.route[i]
        if pt1 != pt2:
            new_path.add_way_point(path.route[i-1].x)
            pt1 = path.route[i-1]
    new_path.add_way_point(path.route[len(path.route)-1].x)

    return new_path

def SmoothRouth(path):
    # path of corner points.
    new_path = Path()
    delta = 10 #distance to smooth each side of corner
    for i in range(len(path)-1):
        pt = path.route[i]
        next_pt = path.route[i+1]
        # new_path.add_way_point(pt.x)
        new_pt = [(pt.x[0]*2+ next_pt.x[0])/3, (pt.x[1]*2+ next_pt.x[1])/3]
        new_path.add_way_point(new_pt)
        new_pt = [(pt.x[0]+ next_pt.x[0]*2)/3, (pt.x[1]+ next_pt.x[1]*2)/3]
        new_path.add_way_point(new_pt)

    new_path.add_way_point(path.route[len(path.route)-1].x)

    return new_path
        
def GetTrackCost(path):
    cost = 0
    for i in range(len(path)-2):
        pt1 = path.route[i].x
        pt2 = path.route[i+1].x
        pt3 = path.route[i+2].x

        dis = f.get_distance(pt1, pt2) 
        angle = f.get_angle(pt1, pt2, pt3)

        cost += dis + angle ** 2

    return cost

def show_path(path):
    interface = Interface(track, 100)
    interface.show_planned_path(path)

# testing
if __name__ == "__main__":
    track = TrackData()
    track.simple_maze()
   

    # myPlanner = A_StarPathFinder(track)
    # path = myPlanner.run_search(5)

    # print(f"Cost of A*  {GetTrackCost(path)}")
    # # show_path(path)
    # path = ReduceRoute(path)
    # # show_path(path)
    # print(f"Cost of Reduced  {GetTrackCost(path)}")
    # path = SmoothRouth(path)
    # # path = SmoothRouth(path)
    # # show_path(path)
    # print(f"Cost of Smoothed  {GetTrackCost(path)}")

    # myPlanner = A_StarPathFinderModified(track)
    myPlanner = RRT_PathFinder(track)
    path = myPlanner.run_search(5)

    # print(f"Cost of Modified  {GetTrackCost(path)}")
    print(f"Cost of RRT  {GetTrackCost(path)}")
    show_path(path)
    
    
