import numpy as np 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
import csv

import LibFunctions as lib
# from TrajectoryPlanner import MinCurvatureTrajectory, ObsAvoidTraj, ShortestTraj


class MapBase:
    def __init__(self, map_name):
        self.name = map_name

        self.scan_map = None
        self.obs_map = None

        self.track = None
        self.track_pts = None
        self.nvecs = None
        self.ws = None
        self.N = None

        self.start = None
        self.wpts = []

        self.height = None
        self.width = None
        self.resolution = None

        self.crop_x = None
        self.crop_y = None

        self.read_yaml_file()
        self.load_map_csv()

    def read_yaml_file(self, print_out=False):
        file_name = 'maps/' + self.name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

            yaml_file = documents.items()
            if print_out:
                for item, doc in yaml_file:
                    print(item, ":", doc)

        self.yaml_file = dict(yaml_file)

        self.resolution = self.yaml_file['resolution']
        self.start = self.yaml_file['start']
        self.crop_x = self.yaml_file['crop_x']
        self.crop_y = self.yaml_file['crop_y']

    def load_map_csv(self):
        track = []
        filename = 'Maps/' + self.name + ".csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded")

        self.track = track
        self.N = len(track)
        self.track_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]

        self.scan_map = np.load(f'Maps/{self.name}.npy')

        self.width = self.scan_map.shape[1]
        self.height = self.scan_map.shape[0]

    def convert_position(self, pt):
        x = pt[0] / self.resolution
        y =  pt[1] / self.resolution

        return x, y

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.convert_position(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)
        
    def convert_int_position(self, pt):
        x = int(round(np.clip(pt[0] / self.resolution, 0, self.width-2)))
        y = int(round(np.clip(pt[1] / self.resolution, 0, self.height-2)))

        return x, y

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        x, y = self.convert_int_position(x_in)
        if x > self.width or y > self.height:
            return True

        if self.scan_map[y, x]:
            return True
        if self.obs_map[y, x]:
            return True
        return False

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.width])
        plt.ylim([self.height, 0])

        track = self.track
        c_line = track[:, 0:2]
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        cx, cy = self.convert_positions(c_line)
        plt.plot(cx, cy, linewidth=2)
        lx, ly = self.convert_positions(l_line)
        plt.plot(lx, ly, linewidth=1)
        rx, ry = self.convert_positions(r_line)
        plt.plot(rx, ry, linewidth=1)

        if self.wpts is not None:
            xs, ys = [], []
            for pt in self.wpts:
                x, y = self.convert_position(pt)
                # plt.plot(x, y, '+', markersize=14)
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, '--', linewidth=2)

        if self.obs_map is None:
            plt.imshow(self.scan_map)
        else:
            plt.imshow(self.obs_map + self.scan_map)

        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)
        if wait:
            plt.show()









class ForestGenerator(MapBase):
    def __init__(self, map_name='forest'):
        self.name = map_name

        self.f_map = np.zeros((120, 500)).T # same as other maps
        self.track = None

        self.gen_path()

    def gen_path(self, N=60):
        tx = 3 # centre line
        txs = np.ones(N) * tx 
        txs = txs[:, None]
        tys = np.linspace(1, 29, N)
        tys = tys[:, None]

        widths = np.ones((N, 2)) * 2.5

        nvecs = np.array([np.ones(N), np.zeros(N)]).T

        self.track = np.concatenate([txs, tys, nvecs, widths], axis=-1)

    def save_map(self):
        np.save('Maps/forest.npy', self.f_map)  

        filename = 'Maps/' + self.name + '.csv'
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.track)

        print(f"Track Saved in File: {filename}")
        


class MapConverter(MapBase):
    def __init__(self, map_name):
        MapBase.__init__(self, map_name)
        # self.name = map_name
        self.yaml_file = None

        self.dt = None
        self.cline = None
        self.search_space = None

        self.crop_x = None
        self.crop_y = None

    def run_conversion(self, show_map=False):
        self.load_map_pgm()
        self.crop_map()
        self.show_map()
        self.find_centreline()
        self.find_nvecs()
        # self.set_widths()
        self.make_binary()
        self.set_true_widths()
        # self.save_map()
        self.render_map(wait=True)

    def load_map_pgm(self):
        self.read_yaml_file()

        map_file_name = self.yaml_file['image']
        pgm_name = 'maps/' + map_file_name


        with open(pgm_name, 'rb') as f:
            codec = f.readline()

        if codec == b"P2\n":
            self.read_p2(pgm_name)
        elif codec == b'P5\n':
            self.read_p5(pgm_name)
        else:
            raise Exception(f"Incorrect format of PGM: {codec}")

        self.obs_map = np.zeros_like(self.scan_map)
        print(f"Map size: {self.width * self.resolution}, {self.height * self.resolution}")

    def read_p2(self, pgm_name):
        print(f"Reading P2 maps")
        with open(pgm_name, 'r') as f:
            lines = f.readlines()

        # This ignores commented lines
        for l in list(lines):
            if l[0] == '#':
                lines.remove(l)
        # here,it makes sure it is ASCII format (P2)
        codec = lines[0].strip()

        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])

        data = (np.array(data[3:]),(data[1],data[0]),data[2])
        self.width = data[1][1]
        self.height = data[1][0]

        data = np.reshape(data[0],data[1])

        self.scan_map = data
    
    def read_p5(self, pgm_name):
        print(f"Reading P5 maps")
        with open(pgm_name, 'rb') as pgmf:
            assert pgmf.readline() == b'P5\n'
            comment = pgmf.readline()
            # comment = pgmf.readline()
            wh_line = pgmf.readline().split()
            (width, height) = [int(i) for i in wh_line]
            depth = int(pgmf.readline())
            assert depth <= 255

            raster = []
            for y in range(height):
                row = []
                for y in range(width):
                    row.append(ord(pgmf.read(1)))
                raster.append(row)
            
        self.height = height
        self.width = width
        self.scan_map = np.array(raster)        

    def convert_to_plot(self, pt):
        x = pt[0] / self.resolution
        y =  pt[1] / self.resolution
        # y = self.height - pt[1] / self.resolution

        return x, y
        
    def convert_to_plot_int(self, pt):
        x = int(round(np.clip(pt[0] / self.resolution, 0, self.width-1)))
        y = int(round(np.clip(pt[1] / self.resolution, 0, self.height-1)))

        return x, y

    def find_centreline(self, show=False):
        self.dt = ndimage.distance_transform_edt(self.scan_map)
        dt = np.array(self.dt) 

        d_search = 1 
        n_search = 11
        dth = (np.pi * 4/5) / (n_search-1)

        # makes a list of search locations
        search_list = []
        for i in range(n_search):
            th = -np.pi/2 + dth * i
            x = -np.sin(th) * d_search
            y = np.cos(th) * d_search
            loc = [x, y]
            search_list.append(loc)

        pt = self.start
        self.cline = [pt]
        th = np.pi/2 # start theta
        while lib.get_distance(pt, self.start) > d_search or len(self.cline) < 10:
            vals = []
            self.search_space = []
            for i in range(n_search):
                d_loc = lib.transform_coords(search_list[i], -th)
                search_loc = lib.add_locations(pt, d_loc)

                self.search_space.append(search_loc)

                x, y = self.convert_to_plot_int(search_loc)
                val = dt[y, x]
                vals.append(val)

            ind = np.argmax(vals)
            d_loc = lib.transform_coords(search_list[ind], -th)
            pt = lib.add_locations(pt, d_loc)
            self.cline.append(pt)

            if show:
                self.plot_raceline_finding()

            th = lib.get_bearing(self.cline[-2], pt)
            print(f"Adding pt: {pt}")

        self.cline = np.array(self.cline)
        self.N = len(self.cline)
        print(f"Raceline found")
        self.plot_raceline_finding(True)

    def find_nvecs(self):
        N = self.N
        track = self.cline

        nvecs = []
        # new_track.append(track[0, :])
        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[0, :], track[1, :]))
        nvecs.append(nvec)
        for i in range(1, len(track)-1):
            pt1 = track[i-1]
            pt2 = track[min((i, N)), :]
            pt3 = track[min((i+1, N-1)), :]

            th1 = lib.get_bearing(pt1, pt2)
            th2 = lib.get_bearing(pt2, pt3)
            if th1 == th2:
                th = th1
            else:
                dth = lib.sub_angles_complex(th1, th2) / 2
                th = lib.add_angles_complex(th2, dth)

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
            nvecs.append(nvec)

        nvec = lib.theta_to_xy(np.pi/2 + lib.get_bearing(track[-2, :], track[-1, :]))
        nvecs.append(nvec)

        self.track = np.concatenate([track, nvecs], axis=-1)

    def set_widths(self, width =0.6):
        track = self.track
        N = len(track)
        ths = [lib.get_bearing(track[i, 0:2], track[i+1, 0:2]) for i in range(N-1)]

        ls, rs = [width], [width]
        for i in range(N-2):
            dth = lib.sub_angles_complex(ths[i+1], ths[i])
            dw = dth / (np.pi) * width
            l = width #+  dw
            r = width #- dw
            ls.append(l)
            rs.append(r)

        ls.append(width)
        rs.append(width)

        ls = np.array(ls)
        rs = np.array(rs)

        new_track = np.concatenate([track, ls[:, None], rs[:, None]], axis=-1)

        self.track = new_track

    def set_true_widths(self):
        nvecs = self.track[:, 2:4]
        tx = self.track[:, 0]
        ty = self.track[:, 1]

        stp_sze = 0.1
        sf = 0.5 # safety factor
        nws, pws = [], []
        for i in range(self.N):
            pt = [tx[i], ty[i]]
            nvec = nvecs[i]

            j = stp_sze
            s_pt = s_pt = lib.add_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = s_pt = lib.sub_locations(pt, nvec, j)
            while not self.check_scan_location(s_pt):
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)

        nws, pws = np.array(nws), np.array(pws)

        new_track = np.concatenate([self.track, nws[:, None], pws[:, None]], axis=-1)

        self.track = new_track

    def plot_race_line(self, nset=None, wait=False):
        plt.figure(2)
        plt.clf()

        track = self.track
        c_line = track[:, 0:2]
        l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
        r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

        # plt.figure(1)
        plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)
        plt.plot(r_line[:, 0], r_line[:, 1], 'x', markersize=12)

        if nset is not None:
            deviation = np.array([track[:, 2] * nset[:, 0], track[:, 3] * nset[:, 0]]).T
            r_line = track[:, 0:2] + deviation
            plt.plot(r_line[:, 0], r_line[:, 1], linewidth=3)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def plot_raceline_finding(self, wait=False):
        plt.figure(1)
        plt.clf()
        plt.imshow(self.dt)


        for pt in self.cline:
            s_x, s_y = self.convert_to_plot(pt)
            plt.plot(s_x, s_y, '+', markersize=16)

        for pt in self.search_space:
            s_x, s_y = self.convert_to_plot(pt)
            plt.plot(s_x, s_y, 'x', markersize=12)


        plt.pause(0.001)

        if wait:
            plt.show()

    def show_map(self):
        plt.figure(1)
        plt.imshow(self.scan_map)

        sx, sy = self.convert_to_plot(self.start)
        plt.plot(sx, sy, 'x', markersize=20)

        plt.show()

    def crop_map(self):
        x = self.crop_x
        y = self.crop_y
        self.scan_map = self.scan_map[y[0]:y[1], x[0]:x[1]]
        
        self.width = self.scan_map.shape[1]
        self.height =  self.scan_map.shape[0]

        print(f"Map cropped: {self.height}, {self.width}")

    def make_binary(self):
        avg = np.mean(self.scan_map)
        for i in range(self.height):
            for j in range(self.width):
                if self.scan_map[i, j] > avg:
                    self.scan_map[i, j] = False
                else:
                    self.scan_map[i, j] = True 

        self.show_map()
        np.save(f'Maps/{self.name}.npy', self.scan_map)

    def set_map_params(self):
        # this is a function to set the map parameters
        # self.crop_x = [0, -1]
        # self.crop_y = [0, -1]

        self.crop_x = [200, 500]
        self.crop_y = [100, 600]

        self.start = [8, 2.8]
        print(f"start: {self.start}")

        self.yaml_file['start'] = self.start
        self.yaml_file['crop_x'] = self.crop_x
        self.yaml_file['crop_y'] = self.crop_y

        yaml_name = 'maps/' + self.name + '.yaml'
        with open(yaml_name, 'w') as yaml_file:
            yaml.dump(self.yaml_file, yaml_file)

    def save_map(self):
        filename = 'Maps/' + self.name + '.csv'

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.track)

        print(f"Track Saved in File: {filename}")


def test_map_converter():
    names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
    name = names[5]
    myConv = MapConverter(name)
    myConv.run_conversion()

    # t = SimMap(name)
    # t.get_min_curve_path()
    # t.render_map(wait=True)


# def forest_gen():
#     f = ForestGenerator()
#     f.save_map()
