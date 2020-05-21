import numpy as np  
from tkinter import *
from TrackMapData import TrackMapData
from pickle import load, dump

class TrackMapBase:
    def __init__(self, track_obj=None):
        if track_obj is None:
            self.map_data = TrackMapData()
        else:
            self.map_data = track_obj

        size = self.map_data.display_size

        self.rect_map = np.zeros_like(self.map_data.track_map, dtype=np.int)

        self.root = Tk()
        frame = Frame(self.root, height=size[0], width=size[1])
        frame.grid(row=1, column=1)
        self.canv = Canvas(frame, height=size[0], width=size[1])
        self.save_pane = Frame(self.root, height=size[0], width=size[1]/10)
        self.save_pane.grid(row=1, column=2)

        self.set_up_buttons()

    def redrawmap(self):
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)
                
                idx = self.rect_map[i, j]
                c.itemconfig(idx, fill=color)

    def set_up_buttons(self):
        save_pane = self.save_pane
        quit_button = Button(save_pane, text="Quit", command=self.root.destroy)
        quit_button.pack()

        self.name_var = StringVar()
        self.name_box = Entry(save_pane, text="myTrack0", textvariable=self.name_var)
        self.name_var.set("myTrack0")
        self.name_box.pack()

        save_button = Button(save_pane, text="Save", command=self.save_map)
        save_button.pack()
        save_button = Button(save_pane, text="Load", command=self.load_map)
        save_button.pack()
        clear_button = Button(save_pane, text="Clear", command=self.clear_map)
        clear_button.pack()

        reset_obs = Button(save_pane, text="Reset Obs", comman=self.reset_obs)
        reset_obs.pack()

        add_obs = Button(save_pane, text="Add Obs", command=self.add_obs)
        add_obs.pack()


# button features
    def save_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'wb')
        
        dump(self.map_data, db_file)
        # np.save(filename, self.map)
        print(f"File saved: {filename}")

    def load_map(self, info=None):
        print("Attempting map load")
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'rb')

        load_map = load(db_file)
        db_file.close()
        if type(self.map_data) == type(load_map):
            print("Map loaded successfully")
            self.map_data = load_map
        else:
            print("Problem loading map")
            print(f"Loaded map type: {type(load_map)}")

        self.redrawmap()

    def reset_obs(self):
        self.map_data.reset_obstacles()
        self.redrawmap()

    def add_obs(self):
        self.map_data.add_random_obstacle()
        print(f"Obs Added: {self.map_data.obstacles[-1].size}")
        self.reset_obs()

    def clear_map(self):
        self.map_data = TrackMapData()
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




class TrackMapInterface2(TrackMapBase):
    def __init__(self, track_obj):
        super().__init__(track_obj=track_obj)

        self.create_map()

        self.root.mainloop()

    # set up - define create map and additional buttons
    def create_map(self):
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)

                top_left = (i*block_sz, j*block_sz)
                bot_right = ((i+1)*block_sz, (j+1)*block_sz)
                rect = c.create_rectangle(top_left, bot_right, fill=color)
                self.rect_map[i, j] = rect
                c.pack()


class TrackGenerator2(TrackMapBase):
    def __init__(self):
        super().__init__()

        self.create_map()
        self.root.bind("<Return>", self.set_wp)
        self.root.mainloop()

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
        # print(info)
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

    def set_wp(self, info):
        # print(info)
        x, y = self.get_loaction_value(info.x, info.y)
        x = [x, y]
        # print(x)
        if x in self.map_data.way_pts:
            self.map_data.way_pts.remove(x)
        else:
            self.map_data.way_pts.append(x)
        self.redrawmap()




class TrackMapInterface:
    def __init__(self, track_obj):
        self.map_data = track_obj
        size = self.map_data.display_size

        self.rect_map = np.zeros_like(self.map_data.track_map, dtype=np.int)

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
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
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
        save_button = Button(save_pane, text="Save", command=self.save_map)
        save_button.pack()
        save_button = Button(save_pane, text="Load", command=self.load_map)
        save_button.pack()

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
        # np.save(filename, self.map)

    def load_map(self, info=None):
        print("Attempting map load")
        filename = "DataRecords/" + str(self.name_var.get()) 
        db_file = open(filename, 'rb')

        load_map = load(db_file)
        db_file.close()
        if type(self.map_data) == type(load_map):
            print("Map loaded successfully")
            self.map_data = load_map
        else:
            print("Problem loading map")
            print(f"Loaded map type: {type(load_map)}")

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



def load_map(map_name="myTrack1"):
    filename = "DataRecords/" + map_name 
    db_file = open(filename, 'rb')
    loadmap = load(db_file)

    return loadmap

def test_interface():
    # map_data = load_map()
    # myInterface = TrackMapInterface(map_data)
    # myInterface = TrackMapInterface2(track_obj = map_data)
    myTrackMap = TrackGenerator2()


if __name__ == "__main__":
    test_interface()

