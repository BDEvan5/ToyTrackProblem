import numpy as np 
from tkinter import *


class Node:
    def __init__(x, y):
        self.x = x
        self.y = y


class TrackMap:
    def __init__(self, resolution=2, scaling_factor=10):
        # scale to 100 by 100 in space and 50 by 50 in blocks
        self.fs = int(scaling_factor)
        self.res = int(resolution)
        self.map_size = np.array([100, 100])
        self.size = self.map_size * self.fs
        self.res_size = self.size / resolution

        self.map = np.zeros((int(self.res_size[0]), int(self.res_size[1])), dtype=np.bool) # if it is walkable

        self.root = Tk()

        self.set_up_map()

    def set_up_map(self):
        fs = self.fs # scaling factor for display resolution
        block_sz = self.fs * self.res

        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])
        canv = self.canv

        for i in range(int(self.map_size[0]/self.res)):
            for j in range(int(self.map_size[1]/self.res)):
                if self.map[i, j]:
                    color = 'grey50'
                else:
                    color = 'gray95'
                top_left = (i*block_sz, j*block_sz)
                bot_right = ((i+1)*block_sz, (j+1)*block_sz)
                canv.create_rectangle(top_left, bot_right, fill=color)

                canv.pack()

        self.root.mainloop()

    # def show_map(self):



if __name__ == "__main__":
    myTrackMap = TrackMap()
    
