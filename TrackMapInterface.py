import numpy as np  
import tkinter as tk 
from TrackMapData import TrackMapData
from pickle import load, dump
import LibFunctions as f
import pyautogui 
from StateStructs import SimulationState
import multiprocessing as mp
import sys


class TrackMapBase:
    def __init__(self, track_obj=None):
        if track_obj is None:
            self.map_data = TrackMapData()
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
        self.map_data = TrackMapData()
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

        return color

    def get_loaction_value(self, x, y):
        block_size = self.map_data.fs * self.map_data.res

        x_ret = int(np.floor(x / block_size))
        y_ret = int(np.floor(y / block_size))

        return x_ret, y_ret


class TrackGenerator(TrackMapBase):
    def __init__(self):
        super().__init__()

        self.root.bind("<Return>", self.set_wp)
        self.create_map()
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
    def __init__(self, track_obj, dt=100):
        super().__init__(track_obj=track_obj)

        self.create_map()
        self.set_up_info_pannel()
        self.set_up_extra_buttons()

        self.pause_flag = True # start in b_pause mode
        self.step_i = SimulationState()
        self.step_q = mp.Queue()
        self.range_lines = []
        self.prev_px = [0, 0]
        self.save_shot_path = "DataRecords/EndShot"
        self.dt = dt


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
        self.root.mainloop()

    def setup_root(self):
        # print("Setup root called")
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)

        self.canv.move(self.o, px[0], px[1])
        
        self.canv.pack()
        self.root.after(0, self.run_interface_loop)
        self.run_loop()

    def show_path_setup_root(self, path):
        # print("Setup root called")
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)

        self.canv.move(self.o, px[0], px[1])
        
        self.canv.pack()

        for i, point in enumerate(path.route):
            x = self._scale_input(point.x)
            str_msg = str(i)
            self.end_x = self.canv.create_text(x[0], x[1], text=str_msg, fill='black', font = "Times 20 bold")

            self.canv.pack()   

        self.root.after(0, self.run_interface_loop)
        self.run_loop()

    # main function with logic
    def run_interface_loop(self):
        if  not self.pause_flag:
            self.single_step()
        else:
            self.root.after(self.dt, self.run_interface_loop)

    def show_planned_path(self, path):
        for i, point in enumerate(path):
            new_pt = f.add_locations(point, [0.5, 0.5]) # brings to centre
            x = self._scale_input(new_pt)
            self.end_x = self.canv.create_oval(x[0], x[1], x[0] + 8, x[1] + 8, fill='DeepPink2')
            str_msg = str(i)
            # self.end_x = self.canv.create_text(x[0], x[1], text=str_msg, fill='black', font = "Times 12")

            self.canv.pack() 

        self.run_loop()

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
        if not self.step_q.empty():
            self.step_i = self.step_q.get()
            if self.step_i.env_state.done is False:
                self.update_car_position()
                self.draw_ranges()
                self.update_info()
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
    def update_car_position(self):
        # move dot
        current_pos = self._scale_input(self.step_i.car_state.x)
        px = f.sub_locations(current_pos, self.prev_px)
        self.canv.move(self.o, px[0], px[1])

        # add line segment
        self.canv.create_line(self.prev_px, current_pos, fill='purple', width=4)

        self.prev_px = current_pos

        # direction line
        length = 40
        self.canv.delete(self.th)
        add_coord = [length * np.sin(self.step_i.car_state.theta), -length * np.cos(self.step_i.car_state.theta)]
        new_pos = f.add_locations(self.prev_px, add_coord)
        self.th = self.canv.create_line(self.prev_px, new_pos, fill='green', width=6)

    def update_info(self):
        step = self.step_i.step
        x = np.around(self.step_i.car_state.x, 2)
        v = np.around(self.step_i.car_state.v, 2)
        th = np.around(self.step_i.car_state.theta, 2)
        reward = np.around(self.step_i.env_state.reward, 3)
        action = np.around(self.step_i.env_state.control_action, 2)
        distance = np.around(self.step_i.car_state.cur_distance)
        state_vec = np.around(self.step_i.car_state.get_state_observation(), 2)
        agent_action = np.around(self.step_i.env_state.agent_action)
        crash_indi = np.around(self.step_i.car_state.crash_chance)

        step_text = str(step)
        self.step.config(text=step_text)

        location_text = str(x)
        self.loc.config(text=location_text)
        velocity_text = str(v) + " @ " + str(th)
        self.velocity.config(text=velocity_text)

        reward_text = str(reward)
        self.reward.config(text=reward_text)

        action_text = str(action)
        self.action.config(text=action_text)

        self.distance.config(text=str(distance))

        state_vec_str = str(state_vec[0:2]) + "\n"  + str(state_vec[2:4]) + "\n" \
            + str(state_vec[4:6]) + "\n" + str(state_vec[6::])
        self.state_vec.config(text=state_vec_str)
        self.agent_action.config(text=str(agent_action))

        self.crash_indicator.config(text=str(crash_indi))
        
    def draw_ranges(self):
        for obj in self.range_lines: # deletes old lines
            self.canv.delete(obj)

        x_scale = self._scale_input(self.step_i.car_state.x)
        for ran in self.step_i.car_state.ranges:
            th = ran.angle + self.step_i.car_state.theta
            dx = [ran.val * np.sin(th), - ran.val * np.cos(th)]
            x1 = f.add_locations(dx, self.step_i.car_state.x)

            node = self._scale_input(x1)

            l = self.canv.create_line(x_scale, node, fill='red', width=4)
            self.canv.pack()
            self.range_lines.append(l)

    def take_screenshot(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        arr = [x, y, x+width, y+height]
        # print(arr)
        path = self.save_shot_path + ".png"
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

def render_track_ep(track, path, sim_mem, screen_name_path="DataRecords/PathTracker", pause=False):
    dt = 60
    interface = TrackMapInterface(track, dt)
    interface.save_shot_path = screen_name_path
    interface.pause_flag = pause

    for step in sim_mem.steps:
        # step.print_point("Step Q")
        interface.step_q.put(step)

    interface.show_path_setup_root(path)

if __name__ == "__main__":
    myTrackMap = TrackGenerator()
    # test_interface()

