    
# these are minor helper data structures for the env

class Location:
    def __init__(self, x=[0, 0]):
        self.x = x

    def set_location(self, x):
        self.x = x

    def get_location(self):
        return self.x


class State(Location):
    def __init__(self, v=[0, 0], x=[0, 0]):
        Location.__init__(self, x) # it would appear that you must innitialise an inherited object 
        self.v = v
    
    def update_state(self, dv, dd):
        for i in range(2):
            self.x[i] += dd[i]
            self.v[i] += dv[i]

    def set_state(self, x=[0, 0], v=[0, 0] ):
        self.x = x
        self.v = v
