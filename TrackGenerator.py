import numpy as np 
from tkinter import *
from TrackMapData import TrackMapData
from pickle import load, dump


class TrackGenerator:
    def __init__(self, resolution=2, scaling_factor=10):
        # scale to 100 by 100 in space and 50 by 50 in blocks
        fs = int(scaling_factor)
        res = int(resolution)
        map_size = np.array([100, 100])
        display_size = map_size * fs
        res_size = display_size / resolution
        n_blocks = int(map_size[0]/res)

        track_map = np.zeros((n_blocks, n_blocks), dtype=np.bool) # if it is walkable
        self.map_data = TrackMapData(track_map)
        self.map_data.set_map_parameters(fs, res, display_size, n_blocks, map_size)
        
        self.rect_map = np.zeros_like(track_map, dtype=np.int)

        self.root = Tk()
        frame = Frame(self.root, height=display_size[0], width=display_size[1])
        frame.grid(row=1, column=1)
        self.canv = Canvas(frame, height=display_size[0], width=display_size[1])

        self.create_map()
        self.set_up_saving()

        self.root.mainloop()

# set up
    def create_map(self):
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)

                top_left = (i*block_sz, j*block_sz)
                bot_right = ((i+1)*block_sz, (j+1)*block_sz)
                tag_string = f"button:{i},{j}"
                rect = c.create_rectangle(top_left, bot_right, fill=color, tags=tag_string)
                self.rect_map[i, j] = rect
                c.pack()
                c.tag_bind(tag_string, "<Button-1>", self.set_button_fill)
                c.tag_bind(tag_string, "<B1-Motion>", self.set_button_fill)
                c.tag_bind(tag_string, "<Button-3>", self.set_button_empty)
                c.tag_bind(tag_string, "<B3-Motion>", self.set_button_empty)
                c.tag_bind(tag_string, "<Button-2>", self.set_x)
                # c.tag_bind(tag_string, "e", self.set_x2)

    def set_up_saving(self):
        root = self.root
        # root.bind("<Enter>", self.save_map)
        size = self.map_data.display_size
        save_pane = Frame(root, height=size[0], width=size[1]/10)
        # save_pane.pack(side=RIGHT)
        save_pane.grid(row=1, column=2)

        save_button = Button(save_pane, text="Save", command=self.save_map)
        save_button.pack()
        save_button = Button(save_pane, text="Load", command=self.load_map)
        save_button.pack()

        self.name_var = StringVar()
        self.name_box = Entry(save_pane, text="myTrack0", textvariable=self.name_var)
        self.name_var.set("myTrack0")
        self.name_box.pack()

        clear_button = Button(save_pane, text="Clear", command=self.clear_map)
        clear_button.pack()
        quit_button = Button(save_pane, text="Quit", command=self.root.destroy)
        quit_button.pack()

        reset_obs = Button(save_pane, text="Reset Obs", comman=self.reset_obs)
        reset_obs.pack()

        add_obs = Button(save_pane, text="Add Obs", command=self.add_obs)
        add_obs.pack()

    def redrawmap(self):
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)
                
                idx = self.rect_map[i, j]
                c.itemconfig(idx, fill=color)


#Button features
    def clear_map(self):
        self.map_data.track_map = np.zeros((self.map_data.n_blocks, self.map_data.n_blocks), dtype=np.bool)
        self.map_data.obstacles.clear()
        self.map_data.reset_obstacles()
        self.redrawmap()

    def save_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'ab')
        
        dump(self.map_data, db_file)
        # np.save(filename, self.map)

    def load_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'rb')

        load_map = load(db_file)
        self.map_data = load_map

        self.redrawmap()

    def reset_obs(self):
        self.map_data.reset_obstacles()
        self.redrawmap()

    def add_obs(self):
        self.map_data.add_random_obstacle()
        print(f"Obs Added: {self.map_data.obstacles[-1].size}")
        self.reset_obs()

    def save_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'ab')
        
        dump(self.map_data, db_file)

# bindings
    def set_button_fill(self, info):
        i, j = self.get_loaction_value(info.x, info.y)
        # print(f"Button clicked ON: {info.x};{info.y} --> {i}:{j}")

        self.map_data.track_map[i, j] = True

        color = self.get_map_color(i, j)
        idx = self.rect_map[i, j]    
        self.canv.itemconfig(idx, fill=color)

    def set_button_empty(self, info):
        i, j = self.get_loaction_value(info.x, info.y)
        # print(f"Button clicked OFF: {info.x};{info.y} --> {i}:{j}")

        self.map_data.track_map[i, j] = False

        color = self.get_map_color(i, j)
        idx = self.rect_map[i, j]    
        self.canv.itemconfig(idx, fill=color)

    def set_x(self, info):
        print(info)
        i, j = self.get_loaction_value(info.x, info.y)
        if self.map_data.start_x1 is None:
            self.map_data.start_x1 = [i, j]
        elif self.map_data.start_x2 is None:
            if j == self.map_data.start_x1[1]:
                self.map_data.start_x2 = [i, j]
        else: 
            self.map_data.start_x1 = [i, j]
            self.map_data.start_x2 = None
        self.map_data.set_start_line()
        self.redrawmap()


# helpers
    def get_map_color(self, i, j):
        if self.map_data.track_map[i, j]:
            color = 'grey50'
        else:
            color = 'gray95'

        if self.map_data.obs_map[i, j]:
            color = 'purple1'

        if [i, j] == self.map_data.start_x1 or [i, j] == self.map_data.start_x2:
            color = 'medium spring green'
        elif [i, j] in self.map_data.start_line:
            color = 'spring green'

        return color

    def get_loaction_value(self, x, y):
        block_size = self.map_data.fs * self.map_data.res

        x_ret = int(np.floor(x / block_size))
        y_ret = int(np.floor(y / block_size))

        return x_ret, y_ret





    


        


    # def show_map(self):



if __name__ == "__main__":
    myTrackMap = TrackGenerator()
    
