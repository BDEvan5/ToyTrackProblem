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
        path = self.get_path_list()

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
                
    def get_path_list(self):
        # this sets the path inside the track to the a star path
        curr = self.current_node
        pos_list = []
        while curr is not None:
            pos_list.append(curr.position)
            
            curr = curr.parent
        pos_list = pos_list[::-1]

        pos_list.append(self.track.end_location)

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




"""
Copied algorithms
"""
# TODO: convert this to a class that can be created and called.
class Line():
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def nearest(G, vex, obstacles, radius, track):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):

        if track.check_line_collision(v, vex):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx

def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex

class Graph:
# ''' Define graph '''
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
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

    def randomPosition(self):
        rx = np.random.random()
        ry = np.random.random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return posx, posy

def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize, track):
#   ''' RRT star algorithm '''
    #redefine functions
    


    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        # if isInObstacle(randvex, obstacles, radius):
        #     continue
        if track._check_collision(randvex):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius, track)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize) 
        # what is happenening here is that he is taking a step in the direction of the randvex
        # that is why there is a near const step size.

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        plot(G)

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            # if isThruObstacle(line, obstacles, radius):
            #     continue
            if track.check_line_collision(vex, newvex):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True
            #print('success')
            # break

    path = dijkstra(G)
    plot(G, path)
    return path


def dijkstra(G):
#   '''
#   Dijkstra algorithm for finding shortest path from start position to end.
#   '''
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

class RTT_StarPathFinder:
    def __init__(self, track):
        self.track = track
        
        start = tuple(track.start_location)
        end = tuple(track.end_location)
        self.G = Graph(start, end)

        self.step_size = 15
        self.radius = 5

    def run_search(self, iterations=800):
        track, G, radius, step_size = self.track, self.G, self.radius, self.step_size

        for _ in range(iterations):
            randvex = G.randomPosition()

            if track._check_collision(randvex):
                continue

            # make into adding function
            nearvex, nearidx = nearest(G, randvex, [], radius, track)
            if nearvex is None:
                continue

            newvex = newVertex(randvex, nearvex, step_size) 

            newidx = G.add_vex(newvex)
            dist = distance(newvex, nearvex)
            G.add_edge(newidx, nearidx, dist)
            G.distances[newidx] = G.distances[nearidx] + dist

            self.update_vertices(newvex, newidx)

            self.check_end(newvex, newidx)

        path = dijkstra(G)
        plot(G, path)
        return path

    def check_end(self, newvex, newidx):
        G, radius = self.G, self.radius
    
        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True
    
    def update_vertices(self, newvex, newidx):
        G, radius, track = self.G, self.radius, self.track
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            # line = Line(vex, newvex)

            if track.check_line_collision(vex, newvex):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

            



def plot_path(path=None):
    fig, ax = plt.subplots()

    paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
    lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
    ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path

def plot(G, path=None):
    plt.figure(2)
    plt.clf()
    ax = plt.gca()
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    # fig, ax = plt.subplots()

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
    plt.pause(0.001)
    # plt.show()



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

    def __iter__(self): #todo: fix this
        for obj in self.route:
            return obj

    def show(self, track=None):
        if track is None:
            track = TrackData()
            track.simple_maze()
        interface = Interface(track, 100)
        interface.show_planned_path(self)


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

# helper for smoother
def get_practice_path():
    track = TrackData()
    track.simple_maze()
   
    myPlanner = A_StarPathFinder(track)
    path = myPlanner.run_search(5)

    return path

def convert_list_to_path(path):
    new_path = Path()
    for pt in path:
        new_path.add_way_point(pt)

    return new_path

def show_path(path):
    track = TrackData()
    track.simple_maze()
    p = Path()
    if type(path) != type(p):
        path = convert_list_to_path(path)

    interface = Interface(track, 100)
    interface.show_planned_path(path)


# define unit tests

def test_a_star():
    track = TrackData()
    track.simple_maze()
   
    myPlanner = A_StarPathFinder(track)
    path = myPlanner.run_search(5)

    # plot_path(path)
    path = convert_list_to_path(path)
    show_path(path)

def test_rrt_star():
    track = TrackData()
    track.simple_maze()
   
    start = tuple(track.start_location)
    end = tuple(track.end_location)

    path = RRT_star(start, end, [], 800, 5, 15, track)
    # plot_path(path)

def test_rrt_star_class():
    track = TrackData()
    track.simple_maze()

    finder = RTT_StarPathFinder(track)
    path = finder.run_search()
    show_path(path)

# testing
if __name__ == "__main__":

    # test_rrt_star()
    # test_a_star()
    test_rrt_star_class()
    
