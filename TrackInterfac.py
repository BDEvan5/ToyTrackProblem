from tkinter import *
import multiprocessing
import LocationState as ls

class Interface:
    def __init__(self, update_time=0, scaling_factor=10, size=[1000, 1000]):
        self.dt = update_time
        self.root = Tk()
        self.location = ls.Location()
        self.end_l = ls.Location()

        self.q = multiprocessing.Queue()

        self.fs = scaling_factor
        self.size = size

        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])
        self.o = self.canv.create_oval(20, 20, 40, 40, fill='red')

        self.prev_px = []
        
    def setup_root(self):
        print("Setup root called")
        
        self.canv.pack()
        self.root.after(0, self.move_o)
        self.root.mainloop()

    def move_o(self):
        # gets the change in xy and then updates
        px = self.get_xy()
        
        self.canv.move(self.o, px[0], px[1])
        

        # updates current location of car
        # this could be done analytically knowing the change
        coords = self.canv.coords(self.o)
        x_new = [0, 0]
        x_new[0] = (coords[0] + coords[2])/2
        x_new[1] = (coords[1] + coords[3])/2
        self.location.set_location(x_new)
        self.prev_px.append(x_new)
        self.plot_line()
        self.root.after(self.dt, self.move_o)

    def get_xy(self):
        # _px indicates it is a pixel value
        if not self.q.empty():
            x = self.q.get()
            x = self.scale_input(x)
            px = [0, 0]
            for i  in range(2):
                px[i] = x[i] - self.location.x[i]
        else: 
            px = [0, 0]
        
        return px

    def scale_input(self, x_in):
        x_out = [0, 0] # this creates same size vector
        for i in range(2):
            x_out[i] = x_in[i] * self.fs
        return x_out
        
    def set_end_location(self, x):
        x = self.scale_input(x)
        self.end_l.x = x
        self.end_x = self.canv.create_text(x[0], x[1], text='X')
        self.canv.pack()

    def add_obstacle(self, obs):
        o1 = self.scale_input(obs[0:2])
        o2 = self.scale_input(obs[2:4])
        self.canv.create_rectangle(o1, o2, fill='blue')
        self.canv.pack()

    def plot_line(self):
        i = len(self.prev_px)
        print(i)
        # print(self.prev_px[i-1])
        if i > 2:
            self.canv.create_line(self.prev_px[i-1], self.prev_px[i-2], fill='green')
