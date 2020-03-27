from tkinter import *
import multiprocessing as mp
from StateStructs import SimMem, SimulationState
import LibFunctions as f
import time
import numpy as np 
import pyautogui 
import RaceEnv
import os




class Interface:
    def __init__(self, track, update_time, scaling_factor=10):
        self.dt = update_time
        self.fs = scaling_factor
        self.size = [100*self.fs, 100*self.fs]  # the grid is 100 points scaled
        self.track = track

        self.root = Tk()

        self.step_i = SimulationState()

        self.step_q = mp.Queue()
        self.node_q = mp.Queue()
        self.sense_blocks = []
        self.range_lines = []
        self.prev_px = [0, 0]

        self.pause_flag = False

        self.setup_window()  
        self.set_up_track()   
        self.set_up_buttons()

    def setup_window(self):
        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])

        p1 = [30, 30]
        p2 = [70, 70]
        self.o = self.canv.create_oval(p1, p2, fill='red')
        self.th = self.canv.create_line(50, 50, 20, 20, fill='green', width=4) 


        self.info_p = Frame(self.root, height=self.size[0]/2, width=(self.size[1]/10))
        self.info_p.pack(side= RIGHT)

        self.step_name = Label(self.info_p, text="Step")
        self.step_name.pack()
        self.step = Label(self.info_p, text=0)
        self.step.pack()

        self.location_name = Label(self.info_p, text="Location")
        self.location_name.pack()
        self.loc = Label(self.info_p, text=(0, 0))
        self.loc.pack()

        self.velocity_name = Label(self.info_p, text="Velocity")
        self.velocity_name.pack()
        self.velocity = Label(self.info_p, text="0 @ 0")
        self.velocity.pack()

        # this is a canvas to hold the senses
        self.sense_name = Label(self.info_p, text="Sensing")
        self.sense_name.pack()
        self.sense_canv = Canvas(self.info_p, height=100, width=100)
        self.sense_canv.pack(side=BOTTOM)
        
        for i in range(3):
            for j in range(3):
                p1 = (33*j, 33*i)
                p2 = (33*j+30, 33*i + 30)
                s = self.sense_canv.create_rectangle(p1, p2, fill='black')
                self.sense_blocks.append(s)

        self.reward_name = Label(self.info_p, text="Reward")
        self.reward_name.pack()
        self.reward = Label(self.info_p, text="0")
        self.reward.pack()

        self.action_name = Label(self.info_p, text="Action")
        self.action_name.pack()
        self.action = Label(self.info_p, text="[0, 0]")
        self.action.pack()

        self.distance_name = Label(self.info_p, text="Distance To Target")
        self.distance_name.pack()
        self.distance = Label(self.info_p, text="0")
        self.distance.pack()
        
    def set_up_buttons(self):
        self.b = Frame(self.info_p, height=self.size[0]/2, width=(self.size[1]/10))
        self.b.pack(side= BOTTOM)

        self.pause = Button(self.b, text='Pause', command=self.pause_set)
        self.pause.pack()
        
        self.play = Button(self.b, text="Play", command=self.play_set)
        self.play.pack()

        # self.quit = Button(self.b, text='Quit', command=)
        # self.quit.pack()

    def pause_set(self):
        self.pause_flag = True

    def play_set(self):
        self.pause_flag = False

    def set_up_track(self):
        # self.canv.create_rectangle([0, 0], self.size, fill='blue')
        # self.canv.pack()
        # self.canv.create_rectangle(self.track.boundary, fill='white')
        # self.canv.pack()

        for obs in self.track.obstacles:
            o1 = self._scale_input(obs[0:2])
            o2 = self._scale_input(obs[2:4])
            self.canv.create_rectangle(o1, o2, fill='blue')
            self.canv.pack()

        for obs in self.track.hidden_obstacles:
            o1 = self._scale_input(obs[0:2])
            o2 = self._scale_input(obs[2:4])
            self.canv.create_rectangle(o1, o2, fill='blue')
            self.canv.pack()

        x = self._scale_input(self.track.end_location)
        self.end_x = self.canv.create_text(x[0], x[1], text='X', fill='brown', font = "Times 20 italic bold")
        self.canv.pack()    

        self.prev_px = self._scale_input(self.track.start_location)   

        for i, point in enumerate(self.track.route):
            # print(point)
            x = self._scale_input(point.x)
            str_msg = str(i)
            # self.end_x = self.canv.create_text(x[0], x[1], text="-", fill='black', font = "Times 20 bold")
            self.end_x = self.canv.create_text(x[0], x[1], text=str_msg, fill='black', font = "Times 20 bold")

            self.canv.pack()   

    def setup_root(self):
        print("Setup root called")
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)

        self.canv.move(self.o, px[0], px[1])
        
        self.canv.pack()
        self.root.after(0, self.run_interface_loop)
        self.root.mainloop()

    def run_interface_loop(self):
        if  not self.pause_flag:
            self.get_step_info()
            if self.step_i.env_state.done is False:
                self.update_position()
                # self.draw_ranges()
                self.update_info()
                self.root.after(self.dt, self.run_interface_loop)
            else:
                print("Going to destroy tk inter")
                self.take_screenshot()
                self.root.destroy()
        else:
            self.root.after(self.dt, self.run_interface_loop)

    def _scale_input(self, x_in):
        x_out = [0, 0] # this creates same size vector
        for i in range(2):
            x_out[i] = x_in[i] * self.fs
        return x_out
  
    def update_position(self):
        current_pos = self._scale_input(self.step_i.car_state.x)
        # print("Current: " + str(current_pos) + " -> Prev: " + str(self.prev_px))

        px = f.sub_locations(current_pos, self.prev_px)

        self.canv.move(self.o, px[0], px[1])
        self.canv.create_line(self.prev_px, current_pos, fill='purple', width=4)

        self.prev_px = current_pos

        length = 40
        self.canv.delete(self.th)
        add_coord = [length * np.sin(self.step_i.car_state.theta), -length * np.cos(self.step_i.car_state.theta)]
        new_pos = f.add_locations(self.prev_px, add_coord)
        self.th = self.canv.create_line(self.prev_px, new_pos, fill='green', width=6)

    def update_info(self):
        step_text = str(self.step_i.step)
        self.step.config(text=step_text)

        location_text = str(np.around(self.step_i.car_state.x, 3))
        self.loc.config(text=location_text)
        velocity_text = str(self.step_i.car_state.v) + " @ " + str(self.step_i.car_state.theta)
        self.velocity.config(text=velocity_text)

        reward_text = str(self.step_i.env_state.reward)
        self.reward.config(text=reward_text)

        action_text = str(np.around(self.step_i.env_state.action, 2))
        self.action.config(text=action_text)

        self.distance.config(text=str(self.step_i.env_state.distance_to_target))

    def get_step_info(self):
        self.step_i = self.step_q.get()
        

        # self.step_i.print_step()
        
    def show_map(self):
        print("Show Map called")
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)

        self.canv.move(self.o, px[0], px[1])
        
        self.canv.pack()

        self.root.mainloop()

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

    def take_screenshot(self, screen_name="RunFiles/end_shot.png"):
        path = screen_name
        pyautogui.screenshot(path)


class InterfaceManager:
    def __init__(self, track, dt):
        self.ep = SimMem()
        self.interface = None
        self.dt = dt
        self.track = track

    def run_replay(self, ep_mem):
        #Creates new interface for each ep that is played
        self.interface = Interface(self.track, self.dt)
        self.ep = ep_mem

        # print("Starting Replay")
        for step in self.ep.steps:
            self.interface.step_q.put(step)

        self.interface.setup_root()

    def replay_last(self):
        self.ep.load_ep("RunFiles/Last_ep")
        self.run_replay(self.ep)

    def replay_best(self):
        self.ep.load_ep("RunFiles/BestRun")
        self.run_replay(self.ep)

    def replay_ep(self, file_name):
        try:
            self.ep.load_ep(file_name)
            self.run_replay(self.ep)
        except Exception as e:
            print("File could not be opened: " + str(file_name))
            print(e)
        
        

    def show_route(self): # to be removed in future versions.
        root = mp.Process(target=self.interface.show_map)
        root.start()
        time.sleep(10)
        root.terminate()

    def replay_tests(self):
        
        for file_name in os.listdir(os.getcwd() + "/SimulationTests/"):
            self.replay_ep("SimulationTests/" + file_name)




