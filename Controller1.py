import numpy as np 
import TrackEnv1
import LocationState as ls
import time
import logging
import LibFunctions as f


class RandomAgent:
    # this class controlls the car
    def __init__(self, env):
        self.env = env

    def random_agent(self, steps=200):
        print("Random agent called")
        action_range = 1

        ep_reward = 0
        state = self.env.reset() # not needed until learning starts
        for t in range(steps):
            action = [np.random.randn(), np.random.randn()]

            state, reward, done = self.env.step(action)
            self.env.render() 

            if done:
                break
        print("Random Episode done")

class OptimalAgent:
    def __init__(self, env, logger, sleep=0.1, fa=1):
        self.env = env 
        self.sleep = sleep
        self.fa = fa # action factor for how much to relate action to difference
        self.end = ls.Location()
        self.start = ls.Location()

        self.start_n = None
        self.end_n = None

        self.open_list = []
        self.closed_list = []
        self.current_node = None

        self.logger = logger

    def set_locations(self, x_start, x_end):
        self.start = x_start
        self.end = x_end
        self.env.add_locations(x_start, x_end)

        self.start_n = Node(None, x_start)
        self.end_n = Node(None, x_end)

        self.open_list.append(self.start_n)

    def opti_agent(self, steps=50):
        print("Optimial agent called")

        state = self.env.reset() # not needed until learning starts
        done = False
        self.logger.debug("0: ----------------------" )
        self.logger.debug("New State - " + str(state.x))

        step = 1
        while len(self.open_list) > 0 and step < steps:
            self.next_node = Node(position=state.x, parent=self.current_node)
            self.current_node = self.next_node
      
            if done:
                break

            action = self.select_action()
            
            state, _, done = self.env.step(action)
            self.env.render() 

            self.logger.debug("%d: ----------------------" %(step))
            self.logger.debug("Action taken - " + str(action))
            self.logger.debug("New State - " + str(state.x))

            time.sleep(self.sleep)
            
            step += 1
        
        print("Optimal Episode done in %d steps" % step)

    def calc_open_list(self):
        # this function calculates the surrounding node values
        dx = self.env.dx / 10
        children = []
        for new_position in [(0, -dx), (0, dx), (-dx, 0), (dx, 0), (-dx, -dx), (-dx, dx), (dx, -dx), (dx, dx)]: # Adjacent squares

            # Get node position
            node_position = (self.current_node.position[0] + new_position[0], self.current_node.position[1] + new_position[1])

            # Create new node
            new_node = Node(self.current_node, node_position)

            # Create the f, g, and h values
            new_node.g = self.current_node.g + 1
            new_node.h = ((new_node.position[0] - self.end_n.position[0]) ** 2) + ((new_node.position[1] - self.end_n.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            # Append
            children.append(new_node)

        return children
        
    def select_action(self):
        children = self.calc_open_list()

        current_node = children[0]
        for option in children:
            if option.f < current_node.f:
                current_node = option

        action = [0.0, 0.0]
        for i in range(2):
            action[i] = self.end_n.position[i] - current_node.position[i] *self.fa

        return action

    def random_agent(self, steps=200):
        # this is just here to test
        print("Random agent called")
        action_range = 1

        ep_reward = 0
        state = self.env.reset() # not needed until learning starts
        for t in range(steps):
            action = [np.random.randn(), np.random.randn()]

            state, reward, done = self.env.step(action)
            self.env.render() 

            if done:
                break
        print("Random Episode done")


class StarAOpti:
    def __init__(self, env, logger, sleep=0.4, fa=5):
        self.env = env 
        self.sleep = sleep
        self.fa = fa # action factor for how much to relate action to difference
        self.end = ls.Location()
        self.start = ls.Location()

        self.start_n = None
        self.end_n = None
        self.state = None

        self.open_list = []
        self.closed_list = []
        self.children = []
        self.current_node = None

        self.logger = logger

    def set_locations(self, x_start, x_end):
        self.start = x_start
        self.end = x_end
        self.env.add_locations(x_start, x_end)

        self.start_n = Node(None, x_start)
        self.end_n = Node(None, x_end)

        self.open_list.append(self.start_n)

    def StarA(self, steps=50):
        print("A Star agent called")

        self.find_next_node()

        self.state = self.env.reset() # not needed until learning starts
        done = False
        self.logger.debug("0: ----------------------" )
        self.logger.debug("New State - " + str(self.state.x))

        step = 1
        # while len(self.open_list) > 0 and step < steps:
        while step < steps:
            self.generate_children()

            self.create_child_nodes()

            self.find_next_node()

            action = self.select_action()

            self.state, _, done = self.env.step(action)
            self.env.render() 

            self.update_current_node()
      
            if done:
                break

            self.logger.debug("%d: ----------------------" %(step))
            self.logger.debug("Action taken - " + str(action))
            self.logger.debug("New State - " + str(self.state.x))

            time.sleep(self.sleep)
            
            step += 1
        
        print("Optimal Episode done in %d steps" % step)
    
    def update_current_node(self):
        self.current_node.position = self.state.x

    def find_next_node(self):
        # Get the current node
        self.current_node = self.open_list[0]
        current_index = 0
        for index, item in enumerate(self.open_list):
            if item.f < self.current_node.f:
                self.current_node = item
                current_index = index
        # Pop current off open list, add to closed list
        self.open_list.pop(current_index)
        self.closed_list.append(self.current_node)

    def generate_children(self):
        self.children.clear() # deletes old children

        position_list = []
        for i in range(3):
            for j in range(3):
                dir = [(j-1)*self.env.ds, (i-1)*self.env.ds]
                position_list.append(dir)

        for sense in self.state.senses: # Adjacent squares
            # Get node position
            node_position = sense.sense_location

            if sense.val == 0:

                # Create new node - no obstacle
                new_node = Node(self.current_node, node_position)

                # Create the f, g, and h values
                new_node.g = self.current_node.g + 1
                new_node.h = ((new_node.position[0] - self.end_n.position[0]) ** 2) + ((new_node.position[1] - self.end_n.position[1]) ** 2)
                new_node.f = new_node.g + new_node.h

                # Append
                self.children.append(new_node)

    def create_child_nodes(self):
        for child in self.children:
             # Child is on the closed list
            for closed_child in self.closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = self.current_node.g + 1
            child.h = ((child.position[0] - self.end_n.position[0]) ** 2) + ((child.position[1] - self.end_n.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in self.open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            self.open_list.append(child)
            self.env.track.node_q.put(child.position)

    def select_action(self):
        # this method selects an action to move toward the supposed best node

        # good question is how to know how far to step
        action = [0.0, 0.0]
        dis = f.get_distance(self.end_n.position, self.state.x)
        scale = self.fa * (dis * 0.02)
        for i in range(2):
            action[i] = (self.current_node.position[i] - self.state.x[i]) * scale
        return action

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
