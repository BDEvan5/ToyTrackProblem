from tkinter import *
import multiprocessing as mp
from StateStructs import SimMem, SimulationState
import LibFunctions as f
import time
import numpy as np 
import pyautogui 

from Agent import Model


class NetworkGUI:
    def __init__(self, n_ranges, update_time, scaling_factor=10):
        self.fs = scaling_factor
        self.state = ModelState(n_ranges)
        self.dt = update_time
        self.actions = 3

        self.size = [40*self.fs, 40*self.fs]

        self.root = Tk()
        self.range_lines = []

        action_sapce = 3
        self.model = Model(action_sapce)
        self.action_vals = []

        self.setup_window()

    def load_model(self):
        path = "ModelWeights/target_weights"
        self.model.load_weights(path)

    def setup_window(self):
        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])
        self.canv.pack()

        p1 = [180, 280]
        p2 = [220, 320]
        self.o = self.canv.create_oval(p1, p2, fill='red')
        self.th = self.canv.create_line(200, 300, 200, 250, fill='green', width=4) 

        self.info_p = Frame(self.root, height=20, width=(40))
        self.info_p.pack(side= BOTTOM)

        self.output_p = Frame(self.root, height=40, width=(20))
        self.output_p.pack(side= RIGHT)

        self.b_update = Button(self.info_p, text="Update", command=self.update_model)
        self.b_update.pack()


        self.range_boxes = []
        for i in range(self.state.n_ranges):
            label = Label(self.info_p, text="Range: %d"%i)
            label.pack()
            box = Spinbox(self.info_p, from_=0, to=100)
            box.pack()
            self.range_boxes.append(box)

        self.val_labels = []
        for i in range(self.actions):
            val_lab = Label(self.output_p, text="Action Value: %d"%i)
            val_lab.pack()
            self.val_labels.append(val_lab)

        v_text = Label(self.info_p, text="Velocity")
        v_text.pack()
        self.v_box = Spinbox(self.info_p, from_=-5, to=5, increment=0.1)
        self.v_box.pack()
        th_text = Label(self.info_p, text='Theta')
        th_text.pack()
        self.th_box = Spinbox(self.info_p, from_=-1.5, to=1.5, increment=0.1)
        self.th_box.pack()

    def setup_root(self):
        self.load_model()
        # self.root.after(0, self.run_interface_loop)
        self.root.mainloop()

    # def run_interface_loop(self):
    #     self.root.after(self.dt, self.run_interface_loop)

    def update_model(self):
        self.read_values()
        self.draw_ranges()
        self.predict_actions()
        self.show_actions()

    def draw_ranges(self):
        for obj in self.range_lines: # deletes old lines
            self.canv.delete(obj)

        x_scale = self._scale_input(self.state.x)
        for i, val in enumerate(self.state.range_vals):
            th = self.state.angle * i - np.pi/2 + self.state.th
            # print("Value")
            # print(val)
            # print(th)
            dx = [val * np.sin(th), - val * np.cos(th)]
            x1 = f.add_locations(dx, self.state.x)

            node = self._scale_input(x1)

            l = self.canv.create_line(x_scale, node, fill='red', width=4)
            self.canv.pack()
            self.range_lines.append(l)

    def _scale_input(self, x_in):
        x_out = [0, 0] # this creates same size vector
        for i in range(2):
            x_out[i] = x_in[i] * self.fs
        return x_out
  
    def read_values(self):
        for i in range(self.state.n_ranges):
            self.state.range_vals[i] = int(self.range_boxes[i].get())
            # print(self.state.range_vals[i])

        self.state.v = float(self.v_box.get())
        self.state.th = float(self.th_box.get())

    def predict_actions(self):
        state = self.state.get_state_vector()
        action_vals = self.model.predict_on_batch(state[None, :])
        self.action_vals = action_vals[0, :]

    def show_actions(self):
        # print(self.action_vals)
        for i, val in enumerate(self.action_vals):
            string = "Action: %d -->" %i + str(np.around(val, 3))
            self.val_labels[i].config(text=string)

class ModelState:
    def __init__(self, n_ranges):
        self.x = [20.0, 30.0]
        self.v = 0.0
        self.th = 0.0

        self.n_ranges = n_ranges
        self.angle = np.pi / (n_ranges - 1)
        self.range_vals = [0 for i in range(n_ranges)]

    def get_state_vector(self):
        max_v = 5
        max_theta = np.pi
        max_range = 100

        state = []
        state.append(self.v/max_v)
        state.append(self.th/max_theta)
        for val in self.range_vals:
            state.append(val/max_range)

        state = np.array(state)
        return state


myNG = NetworkGUI(5, 100, 10)
myNG.setup_root()
