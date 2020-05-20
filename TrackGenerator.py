import numpy as np 
from tkinter import *


class TrackMap:
    def __init__(self, resolution=2, scaling_factor=10):
        # scale to 100 by 100 in space and 50 by 50 in blocks
        self.fs = int(scaling_factor)
        self.res = int(resolution)
        self.map_size = np.array([100, 100])
        self.size = self.map_size * self.fs
        self.res_size = self.size / resolution
        self.n_blocks = int(self.map_size[0]/self.res)

        self.map = np.zeros((self.n_blocks, self.n_blocks), dtype=np.bool) # if it is walkable
        self.rect_map = np.zeros_like(self.map, dtype=np.int)

        self.root = Tk()
        frame = Frame(self.root, height=self.size[0], width=self.size[1])
        frame.grid(row=1, column=1)
        self.canv = Canvas(frame, height=self.size[0], width=self.size[1])

        self.draw_map()
        self.set_up_saving()

        self.root.mainloop()

    def draw_map(self):
        block_sz = self.fs * self.res
        c = self.canv

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
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

    def redrawmap(self):
        block_sz = self.fs * self.res
        c = self.canv

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                color = self.get_map_color(i, j)
                
                idx = self.rect_map[i, j]
                c.itemconfig(idx, fill=color)

        

    def set_up_saving(self):
        root = self.root
        # root.bind("<Enter>", self.save_map)

        save_pane = Frame(root, height=self.size[0], width=self.size[1]/10)
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

    def clear_map(self):
        self.map = np.zeros((self.n_blocks, self.n_blocks), dtype=np.bool)
        self.redrawmap()


    def save_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) + ".npy"
        np.save(filename, self.map)

    def load_map(self, info=None):
        filename = "DataRecords/" + str(self.name_var.get()) + ".npy"
        load_map = np.load(filename)
        print(load_map)
        self.map = load_map
        self.redrawmap()


    def set_button_fill(self, info):
        i, j = self.get_loaction_value(info.x, info.y)
        print(f"Button clicked: {info.x};{info.y} --> {i}:{j}")

        self.map[i, j] = True

        color = self.get_map_color(i, j)
        idx = self.rect_map[i, j]    
        self.canv.itemconfig(idx, fill=color)

    def set_button_empty(self, info):
        i, j = self.get_loaction_value(info.x, info.y)
        print(f"Button clicked: {info.x};{info.y} --> {i}:{j}")

        self.map[i, j] = False

        color = self.get_map_color(i, j)
        idx = self.rect_map[i, j]    
        self.canv.itemconfig(idx, fill=color)

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




        


    # def show_map(self):



if __name__ == "__main__":
    myTrackMap = TrackMap()
    
