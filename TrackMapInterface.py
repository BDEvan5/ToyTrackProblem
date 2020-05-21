import numpy as np  
from tkinter import *
from TrackMapData import TrackMapData


class TrackMapInterface:
    def __init__(self, track_obj, resolution=2, scaling_factor=10):
        self.track_map = track_obj
        size = self.track_map.display_size

        self.rect_map = np.zeros_like(self.track_map.track_map, dtype=np.int)

        self.root = Tk()
        frame = Frame(self.root, height=size[0], width=size[1])
        frame.grid(row=1, column=1)
        self.canv = Canvas(frame, height=size[0], width=size[1])
        self.save_pane = Frame(self.root, height=size[0], width=size[1]/10)
        self.save_pane.grid(row=1, column=2)

        self.create_map()
        self.set_up_buttons()

        self.root.mainloop()

# set up
    def create_map(self):
        block_sz = self.track_map.fs * self.track_map.res
        c = self.canv

        for i in range(self.track_map.n_blocks):
            for j in range(self.track_map.n_blocks):
                color = self.get_map_color(i, j)

                top_left = (i*block_sz, j*block_sz)
                bot_right = ((i+1)*block_sz, (j+1)*block_sz)
                rect = c.create_rectangle(top_left, bot_right, fill=color)
                self.rect_map[i, j] = rect
                c.pack()

    def set_up_buttons(self):
        save_pane = self.save_pane
        quit_button = Button(save_pane, text="Quit", command=self.root.destroy)
        quit_button.pack()

        reset_obs = Button(save_pane, text="Reset Obs", comman=self.reset_obs)
        reset_obs.pack()

    def redrawmap(self):
        block_sz = self.track_map.fs * self.track_map.res
        c = self.canv

        for i in range(self.track_map.n_blocks):
            for j in range(self.track_map.n_blocks):
                color = self.get_map_color(i, j)
                
                idx = self.rect_map[i, j]
                c.itemconfig(idx, fill=color)


#Button features
    def reset_obs(self):
        self.track_map.reset_obstacles()
        self.redrawmap()

# helpers
    def get_map_color(self, i, j):
        if self.map[i, j]:
            color = 'grey50'
        else:
            color = 'gray95'

        return color

    def get_loaction_value(self, x, y):
        block_size = self.fs * self.res

        x_ret = int(np.floor(x / block_size))
        y_ret = int(np.floor(y / block_size))

        return x_ret, y_ret



def load_map(map_name="myTrack0"):
    filename = "DataRecords/" + map_name + ".npy"
    loadmap = np.load(filename)

    map_data = TrackMapData(loadmap)

    return map_data


if __name__ == "__main__":
    map_data = load_map()

