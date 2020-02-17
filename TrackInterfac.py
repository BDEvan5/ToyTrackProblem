from tkinter import *
import multiprocessing
import LocationState as ls

class Interface:
    def __init__(self, update_time=100):
        self.dt = update_time
        self.root = Tk()
        self.location = ls.Location([100, 100])

        self.q = multiprocessing.Queue()

        self.fs = 200
        self.size = [1000, 1000]
        
    def setup_root(self):
        print("Setup root called")
        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])
        self.o = self.canv.create_oval(20, 20, 40, 40, fill='red')
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
        