from tkinter import *
import multiprocessing as mp
import LocationState as ls
import LibFunctions as f
import EpisodeMem
import time



class Interface:
    def __init__(self, update_time=100, scaling_factor=10, size=[1000, 1000]):
        self.dt = update_time
        self.fs = scaling_factor
        self.size = size

        self.root = Tk()
        self.end_l = ls.Location()

        self.step_i = EpisodeMem.StepInfo()

        self.step_q = mp.Queue()
        self.node_q = mp.Queue()
        self.sense_blocks = []
        self.prev_px = [0, 0]

        self.setup_window()  

        

    def setup_window(self):
        self.canv = Canvas(self.root, height=self.size[0], width=self.size[1])
        size = 40
        # p1 = p2 = [0, 0]
        # p0 = [50, 50]
        # for i in range(2):
        #     p1[i] = p0[i] - size
        #     p2[i] = p0[i] + size
        p1 = [30, 30]
        p2 = [70, 70]
        print(p1, p2)
        self.o = self.canv.create_oval(p1, p2, fill='red')

        self.info_p = Frame(self.root, height=self.size[0], width=(self.size[1]/10))
        self.info_p.pack(side= RIGHT)

        self.step_name = Label(self.info_p, text="Step")
        self.step_name.pack()
        self.step = Label(self.info_p, text=0)
        self.step.pack()

        self.location_name = Label(self.info_p, text="Location")
        self.location_name.pack()
        self.loc = Label(self.info_p, text=(0, 0))
        self.loc.pack()


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
        
    def setup_root(self):
        print("Setup root called")
        p0 = [50, 50]
        px = f.sub_locations(self.prev_px, p0)

        self.canv.move(self.o, px[0], px[1])
        
        self.canv.pack()
        self.root.after(0, self.run_interface_loop)
        self.root.mainloop()

    def run_interface_loop(self):
        self.get_step_info()
        self.update_position()
        # self.draw_nodes()
        self.update_info()
        self.root.after(self.dt, self.run_interface_loop)

    def scale_input(self, x_in):
        x_out = [0, 0] # this creates same size vector
        for i in range(2):
            x_out[i] = x_in[i] * self.fs
        return x_out
        
    def set_locations(self, start, end):
        x = self.scale_input(end)
        self.end_l.x = x
        self.end_x = self.canv.create_text(x[0], x[1], text='X', fill='blue', font = "Times 20 italic bold")
        self.canv.pack()

        self.prev_px = self.scale_input(start)

    def add_obstacle(self, obs):
        o1 = self.scale_input(obs[0:2])
        o2 = self.scale_input(obs[2:4])
        self.canv.create_rectangle(o1, o2, fill='blue')
        self.canv.pack()

    def update_position(self):
        current_pos = self.scale_input(self.step_i.state.x)
        # print("Current: " + str(current_pos) + " -> Prev: " + str(self.prev_px))

        px = f.sub_locations(current_pos, self.prev_px)
        self.canv.move(self.o, px[0], px[1])

        self.canv.create_line(self.prev_px, current_pos, fill='purple')

        self.prev_px = current_pos

    def draw_nodes(self):
        node = [0, 0]
        dx = 5 # scaling factor for sense - should be same as in track
        for sense in self.step_i.state.senses:
            node = f.add_locations(self.step_i.state.x, sense.dir, dx)
            node = self.scale_input(node)
            if sense.val == 0:
                self.canv.create_text(node[0], node[1], text='X', fill='green')
            else:
                self.canv.create_text(node[0], node[1], text='X', fill='red')
            self.canv.pack()

    def update_info(self):
        step_text = str(self.step_i.state.step)
        self.step.config(text=step_text)

        location_text = str(self.step_i.state.x)
        self.loc.config(text=location_text)

        # self.state.print_sense()
        for s, block in zip(self.step_i.state.senses, self.sense_blocks):
            if s.val == 1:
                self.sense_canv.itemconfig(block, fill='black')
            elif s.val == 0:
                self.sense_canv.itemconfig(block, fill='white')

        reward_text = str(self.step_i.state.reward)
        self.reward.config(text=reward_text)

    def get_step_info(self):
        self.step_i = self.step_q.get()
        self.step_i.print_step()
        




class ReplayEpisode:
    def __init__(self, dt=200):
        self.ep = None
        self.interface = Interface(dt)
        self.dt = dt

    def run_replay(self, ep_mem):
        self.ep = ep_mem

        print("Starting Replay")
        # add steps to q
        for step in self.ep.steps:
            # step.print_step()
            self.interface.step_q.put(step)

        root = mp.Process(target=self.interface.setup_root)
        root.start()
        time.sleep(10)
        root.terminate()

        




