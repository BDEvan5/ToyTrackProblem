import numpy as np  
import tkinter as tk 
from TrackMapData import TrackMapData
from pickle import load, dump
import LibFunctions as f
from pyscreenshot import grab
import multiprocessing as mp
import sys



class TrackMapBase:
    def __init__(self, name, track_obj=None):
        if track_obj is None:
            self.map_data = TrackMapData(name)
        else:
            self.map_data = track_obj

        size = self.map_data.display_size

        self.rect_map = np.zeros_like(self.map_data.track_map, dtype=np.int)

        self.root = tk.Tk()
        frame = tk.Frame(self.root, height=size[0], width=size[1])
        frame.grid(row=1, column=1)
        self.canv = tk.Canvas(frame, height=size[0], width=size[1])
        self.save_pane = tk.Frame(self.root, height=size[0], width=size[1]/10)
        self.save_pane.grid(row=1, column=2)
        self.root.bind("<Return>", self.set_wp)
        self.root.bind("s", self.set_start)
        self.root.bind("e", self.set_end)

        self.set_up_buttons()

    def redrawmap(self):
        # block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)
                
                idx = self.rect_map[i, j]
                c.itemconfig(idx, fill=color)

    def set_up_buttons(self):
        save_pane = self.save_pane
        quit_button_all = tk.Button(save_pane, text="Quit all EP's", command=sys.exit)
        quit_button_all.pack()

        quit_button = tk.Button(save_pane, text="Quit", command=self.root.destroy)
        quit_button.pack()

        self.name_var = tk.StringVar()
        self.name_box = tk.Entry(save_pane, text="myTrack0", textvariable=self.name_var)
        self.name_var.set("myTrack0")
        self.name_box.pack()

        save_button = tk.Button(save_pane, text="Save", command=self.save_map)
        save_button.pack()
        save_button = tk.Button(save_pane, text="Load", command=self.load_map)
        save_button.pack()
        clear_button = tk.Button(save_pane, text="Clear", command=self.clear_map)
        clear_button.pack()

        reset_obs = tk.Button(save_pane, text="Reset Obs", comman=self.reset_obs)
        reset_obs.pack()

        add_obs = tk.Button(save_pane, text="Add Obs", command=self.add_obs)
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
        self.map_data = TrackMapData(self.map_data.name)
        self.redrawmap()

    def set_wp(self, info):
        # print(info)
        x = info.x_root - self.root.winfo_x()
        y = info.y_root - self.root.winfo_y()
        # print(x, y)
        x, y = self.get_loaction_value(x, y)
        x = [x, y]
        # print(x)
        if x in self.map_data.way_pts:
            self.map_data.way_pts.remove(x)
        else:
            self.map_data.way_pts.append(x)
        self.redrawmap()

    def set_start(self, event):
        x = event.x_root - self.root.winfo_x()
        y = event.y_root - self.root.winfo_y()

        x, y = self.get_loaction_value(x, y)
        x = [x, y]

        self.map_data.path_start_location = x 
        self.redrawmap()

        print(f"Start Location: {self.map_data.path_start_location}")

    def set_end(self, event):
        x = event.x_root - self.root.winfo_x()
        y = event.y_root - self.root.winfo_y()

        x, y = self.get_loaction_value(x, y)
        x = [x, y]

        self.map_data.path_end_location = x 
        self.redrawmap()
        print(f"End Location: {self.map_data.path_end_location}")



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
        elif [i, j] in self.map_data.way_pts:
            color = 'light sea green'
        elif [i, j] == list(self.map_data.path_start_location) or [i, j] == list(self.map_data.path_end_location):
            color = 'turquoise1'

        return color

    def get_loaction_value(self, x, y):
        block_size = self.map_data.fs * self.map_data.res

        x_ret = int(np.floor(x / block_size))
        y_ret = int(np.floor(y / block_size))

        return x_ret, y_ret


class TrackGenerator(TrackMapBase):
    def __init__(self, name, auto_start=True):
        super().__init__(name)

        # self.root.bind("<Return>", self.set_wp)
        
        self.create_map()
        if auto_start:
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
                c.tag_bind(tag_string, "<B1-Motion>", self.set_thick_brush_fill)
                c.tag_bind(tag_string, "<Button-3>", self.set_button_empty)
                c.tag_bind(tag_string, "<B3-Motion>", self.set_button_empty)
                c.tag_bind(tag_string, "<Button-2>", self.set_x)

    def manual_start(self):
        self.root.mainloop()

    # bindings
    def set_button_fill(self, info):
        i, j = self.get_loaction_value(info.x, info.y)
        # print(f"Button clicked ON: {info.x};{info.y} --> {i}:{j}")

        self.map_data.track_map[i, j] = True

        color = self.get_map_color(i, j)
        idx = self.rect_map[i, j]    
        self.canv.itemconfig(idx, fill=color)

    def set_thick_brush_fill(self, event):
        i, j = self.get_loaction_value(event.x, event.y)
        # print(f"Button clicked ON: {info.x};{info.y} --> {i}:{j}")\
        for x in range(3):
            for y in range(3):
                new_i = i + x - 1
                new_j = j + y - 1
                self.map_data.track_map[new_i, new_j] = True

                color = self.get_map_color(new_i, new_j)
                idx = self.rect_map[new_i, new_j]    
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




class TrackMapInterface(TrackMapBase):
    def __init__(self, track_obj, dt=100, snap=False):
        super().__init__(track_obj.name, track_obj=track_obj)

        self.create_map()
        self.set_up_info_pannel()
        self.set_up_extra_buttons()

        self.pause_flag = True # start in b_pause mode
        self.memory = None
        self.range_lines = []
        self.prev_px = self._scale_input(track_obj.path_start_location) # should fix weird line
        self.save_shot_path = "DataRecords/EndShot"
        self.dt = dt

# snap functionality
    def snap(self, sim_mem, path=None, show=True):
        # function to produce snap view of map.
        # p0 = [50, 50]
        # px = f.sub_locations(self.prev_px, p0)
        # self.canv.move(self.o, px[0], px[1])
        # self.canv.pack()

        if path is not None:
            last_pt = self._scale_input(path.route[0].x)
            for i, point in enumerate(path.route):
                x = self._scale_input(point.x)
                str_msg = str(i)
                # self.end_x = self.canv.create_text(x[0], x[1], text=str_msg, fill='black', font = "Times 20 bold")
                self.canv.create_oval(x[0], x[1], x[0] + 8, x[1] + 8, fill='DeepPink2')
                self.canv.create_line(last_pt, x, fill='IndianRed1', width=2)
                last_pt = x
                self.canv.pack()   
        
        # self.prev_px = self._scale_input(sim_mem.steps[0].car_state.x)
        for step in sim_mem.steps:
            self.draw_snap_step(step) # todo: remove the need for a queu, just have a list


        self.root.after(500, self.root.destroy)
        self.root.mainloop()

    def draw_snap_step(self, step):
        current_pos = self._scale_input(step.car_state.x)

        # add line segment
        self.canv.create_line(self.prev_px, current_pos, fill='purple', width=4)
        x = current_pos
        y = 8 # size of oval
        self.canv.create_oval(x[0], x[1], x[0] + y, x[1] + y, fill='deep sky blue')
        self.prev_px = current_pos

    def take_snap_shot(self, shot_name):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        arr = [x, y, x+width, y+height]
        # print(arr)
        if shot_name is None:
            path = self.save_shot_path + ".png"
        else: 
            path = shot_name + ".png"
        shot = grab()
        shot.show()


# set up - define create map and additional buttons
    def create_map(self):
        block_sz = self.map_data.fs * self.map_data.res
        c = self.canv

        for i in range(self.map_data.n_blocks):
            for j in range(self.map_data.n_blocks):
                color = self.get_map_color(i, j)

                top_left = (i*block_sz, j*block_sz)
                bot_right = ((i+1)*block_sz, (j+1)*block_sz)
                rect = c.create_rectangle(top_left, bot_right, fill=color, outline='grey76')
                self.rect_map[i, j] = rect
                c.pack()

    def set_up_info_pannel(self):
        # self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])

        p1 = [30, 30]
        p2 = [70, 70]
        self.o = self.canv.create_oval(p1, p2, fill='red')
        self.th = self.canv.create_line(50, 50, 20, 20, fill='green', width=4) 

        # self.info_p = Frame(self.root, height=self.size[0]/2, width=(self.size[1]/10))
        # self.info_p.pack(side= RIGHT)
        self.info_p = self.save_pane

        self.step_name = tk.Label(self.info_p, text="Step")
        self.step_name.pack()
        self.step = tk.Label(self.info_p, text=0)
        self.step.pack()

        self.location_name = tk.Label(self.info_p, text="Location")
        self.location_name.pack()
        self.loc = tk.Label(self.info_p, text=(0, 0))
        self.loc.pack()

        self.velocity_name = tk.Label(self.info_p, text="Velocity")
        self.velocity_name.pack()
        self.velocity = tk.Label(self.info_p, text="0 @ 0")
        self.velocity.pack()

        self.action_name = tk.Label(self.info_p, text="Action")
        self.action_name.pack()
        self.action = tk.Label(self.info_p, text="[0, 0]")
        self.action.pack()

        self.distance_name = tk.Label(self.info_p, text="Distance To Target")
        self.distance_name.pack()
        self.distance = tk.Label(self.info_p, text="0")
        self.distance.pack()

        self.state_name = tk.Label(self.info_p, text="State Vector")
        self.state_name.pack()
        self.state_vec = tk.Label(self.info_p, text="0")
        self.state_vec.pack()

        self.agent_name = tk.Label(self.info_p, text="Agent Action")
        self.agent_name.pack()
        self.agent_action = tk.Label(self.info_p, text="0")
        self.agent_action.pack()

        self.reward_name = tk.Label(self.info_p, text="Reward")
        self.reward_name.pack()
        self.reward = tk.Label(self.info_p, text="0")
        self.reward.pack()

        self.crash_indicator_name = tk.Label(self.info_p, text="Crash Indicator")
        self.crash_indicator_name.pack()
        self.crash_indicator = tk.Label(self.info_p, text="0.0")
        self.crash_indicator.pack()

    def set_up_extra_buttons(self):
        # self.b = Frame(self.info_p, height=self.size[0]/2, width=(self.size[1]/10))
        # self.b.pack(side= BOTTOM)
        self.b = self.save_pane

        self.b_pause = tk.Button(self.b, text='Pause', command=self.pause_set)
        self.b_pause.pack()
        
        self.b_play = tk.Button(self.b, text="Play", command=self.play_set)
        self.b_play.pack()

        self.b_step = tk.Button(self.b, text='SingleStep', command=self.single_step) 
        self.b_step.pack()

    
# run
    def run_loop(self):
        self.root.after(0, self.run_interface_loop)
        self.root.mainloop()

    def start_interface_list(self, memory, full_path=None):
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)
        self.canv.move(self.o, px[0], px[1])
        self.canv.pack()

        if full_path is not None:
            for i in range(len(full_path)):
                point = full_path[i, 0:2]
                x = self._scale_input(point)
                str_msg = str(i)
                self.end_x = self.canv.create_text(x[0], x[1], text=str_msg, fill='black', font = "Times 20 bold")
                self.canv.pack()  

        self.memory = memory
        self.draw_start_end()
        self.run_loop()
    
    def draw_start_end(self):
        start = self.map_data.path_start_location
        end = self.map_data.path_end_location
        x = self._scale_input(start)
        self.canv.create_text(x[0], x[1], text='S', fill='orange', font = "Times 20 bold")
        x = self._scale_input(end)
        self.canv.create_text(x[0], x[1], text='E', fill='orange', font = "Times 20 bold")


    # main function with logic
    def run_interface_loop(self):
        if  not self.pause_flag:
            self.single_step()
        else:
            self.root.after(self.dt, self.run_interface_loop)

    def _scale_input(self, x_in):
        x_out = [0, 0]
        for i in range(2):
            x_out[i] = x_in[i] * self.map_data.fs * self.map_data.res
        return x_out 

# button fcns
    def pause_set(self):
        self.pause_flag = True

    def play_set(self):
        self.pause_flag = False

    def single_step(self):
        if len(self.memory) > 0:
            transition = self.memory.pop(0)
            s, a, r, s_p, d = transition

            if d is False:
                th = s[2] # reverse the scale
                self.update_car_position(s[0:2], th)
                self.draw_ranges(s)
                self.update_info(s, r, a)
                self.root.after(self.dt, self.run_interface_loop)
            else:
                print("Going to destroy tk inter: the ep is done")
                # self.take_screenshot()
                self.root.destroy()
        else:
            print("Going to destroy tk inter: empty queue")
            # self.take_screenshot()
            self.root.destroy()

# update functions
    def update_car_position(self, x, theta):
        current_pos = self._scale_input(x)
        # move dot
        px = f.sub_locations(current_pos, self.prev_px)
        self.canv.move(self.o, px[0], px[1])

        # add line segment
        self.canv.create_line(self.prev_px, current_pos, fill='purple', width=4)
        self.prev_px = current_pos

        # direction line
        length = 40
        self.canv.delete(self.th)
        add_coord = [length * np.sin(theta), -length * np.cos(theta)]
        new_pos = f.add_locations(current_pos, add_coord)
        self.th = self.canv.create_line(current_pos, new_pos, fill='green', width=6)

    def update_info(self, s, r, a):
        x = np.around(s[0:2], 2)
        v = np.around(s[2], 2)
        th = np.around(s[3], 2)
        reward = r
        action = np.around(a, 2)
        state_vec = np.around(s, 2)

        location_text = str(x)
        self.loc.config(text=location_text)
        velocity_text = str(v) + " @ " + str(th)
        self.velocity.config(text=velocity_text)

        reward_text = str(reward)
        self.reward.config(text=reward_text)

        action_text = str(action)
        self.action.config(text=action_text)

        state_vec_str = str(state_vec[0:2]) + "\n"  + str(state_vec[2:4]) + "\n" \
            + str(state_vec[4:6]) + "\n" + str(state_vec[6::])
        self.state_vec.config(text=state_vec_str)
        
    def draw_ranges(self, s):
        for obj in self.range_lines: # deletes old lines
            self.canv.delete(obj)

        x = s[0:2]
        theta_car = s[2]
        ranges = s[3:] 

        x_scale = self._scale_input(x)
        n_ranges = 5
        dth = np.pi / (n_ranges-1)
        for i in range(n_ranges):
            th = i * dth + theta_car - np.pi/2
            value = ranges[i] * 100 #unscale
            dx = [value * np.sin(th), - value * np.cos(th)]
            x1 = f.add_locations(dx, x)
            node = self._scale_input(x1)
            l = self.canv.create_line(x_scale, node, fill='red', width=4)
            self.canv.pack()
            self.range_lines.append(l)

    def take_screenshot(self, shot_name=None):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        arr = [x, y, x+width, y+height]
        # print(arr)
        if shot_name is None:
            path = self.save_shot_path + ".png"
        else: 
            path = shot_name + ".png"
        pyautogui.screenshotUtil.screenshot(path, region=arr)
        # pyautogui.screenshot(path, region=arr)




def load_map(map_name="myTrack0"):
    # map_name="myTrack2"
    filename = "DataRecords/" + map_name 
    db_file = open(filename, 'rb')
    loadmap = load(db_file)

    return loadmap

def test_interface():
    map_data = load_map()
    myInterface = TrackMapInterface(map_data)


# externals
def show_track_path(track, path):
    print("Showing track path")
    myTrackInterface = TrackMapInterface(track)
    myTrackInterface.show_planned_path(path)

def render_track_ep(track, path, sim_mem, screen_name_path="DataRecords/PathTracker", pause=False, dt=60):
    interface = TrackMapInterface(track, dt)
    interface.save_shot_path = screen_name_path #todo: move to a setup function
    interface.pause_flag = pause

    interface.start_interface(sim_mem, path)

def snap_track(track, path, sim_mem, screen_name_path="DataRecords/PathTracker"):
    # print(f"ShowingTrackSnap")
    dt = 60
    interface = TrackMapInterface(track, dt)
    interface.save_shot_path = screen_name_path #todo: move to a setup function

    interface.snap(sim_mem, path)
    # print(f"Snapped")

# new functions
def render_ep(track, memory, full_path=None, pause=True):
    interface = TrackMapInterface(track, 100)
    interface.pause_flag = pause

    if full_path is not None:
        interface.start_interface_list(memory, full_path)
    else:
        interface.start_interface_list(memory)

def make_new_map(name):
    print(f"Generating Map: {name}")
    # generate
    myTrackMap = TrackGenerator(name)
    myTrackMap.name_var.set(name)
    myTrackMap.save_map()


if __name__ == "__main__":
    myTrackMap = TrackGenerator()
    # test_interface()

