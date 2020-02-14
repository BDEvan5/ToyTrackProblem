from tkinter import *
import multiprocessing

class Interface:
    def __init__(self):
        self.root = Tk()

        self.x = 100
        self.y = 100

        self.new_x = 100
        self.new_y = 100

        self.q = multiprocessing.Queue()
        

    def setup_root(self):
        print("Setup root called")
        self.canv = Canvas(self.root, height=1000, width=1000)
        self.o = self.canv.create_oval(20, 20, 40, 40, fill='red')
        self.canv.pack()
        self.move_o()
        self.root.after(0, self.move_o)
        # mainloop()
        self.root.mainloop()

    def move_o(self):
        x, y = self.get_xy()

        # print(x, y)
        self.canv.move(self.o, x, y)

        coords = self.canv.coords(self.o)
        self.x = (coords[0] + coords[2])/2
        self.y = (coords[1] + coords[3])/2

        self.root.after(200, self.move_o)

    def get_xy(self):
        if not self.q.empty():
            x = self.q.get()

            self.new_x = x[0]
            self.new_y = x[1]

        x = self.new_x - self.x
        y = self.new_y - self.y
        return x, y


        