    
# these are minor helper data structures for the env
from copy import deepcopy

class Location:
    def __init__(self, x=[0, 0]):
        self.x = x
        self.name = ""

    def set_location(self, x):
        self.x = x

    def get_location(self):
        return self.x

class SingleSense:
    def __init__(self, dir=[0, 0], val=0):
        self.dir = dir
        self.val = val

    def print_sense(self):
        print(str(self.dir) + " --> Val: " + str(self.val))

class Sense:
    def __init__(self):
        self.senses = []
      
        for i in range(3):
            for j in range(3):
                sense = SingleSense([j-1, 1-i])
                self.senses.append(sense)
                
    def print_sense(self):
        for e in self.senses:
            e.print_sense()


class State(Location, Sense):
    def __init__(self, v=[0, 0], x=[0, 0]):
        Location.__init__(self, x) # it would appear that you must innitialise an inherited object 
        Sense.__init__(self)
        self.v = v

        self.step = 0
    
    def update_state(self, dv, dd):
        self.step += 1
        for i in range(2):
            self.x[i] += dd[i]
            self.v[i] += dv[i]

    def set_state(self, x=[0, 0], v=[0, 0] ):
        self.x = x
        self.v = v

