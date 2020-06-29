from matplotlib import pyplot as plt 
import numpy as np 

name00 = 'DataRecords/TrainTrack1000.npy'
name10 = 'DataRecords/TrainTrack1010.npy'

def upscale_o_grid(name):
    array = np.load(name, allow_pickle=True)
    arr_shape = array.shape
    upscale_dim = 1000
    new_map = np.zeros((upscale_dim, upscale_dim))
    block_range = upscale_dim / arr_shape[0]
    for i in range(upscale_dim):
        for j in range(upscale_dim):
            new_map[i, j] = array[int(i/block_range), int(j/block_range)]

    plt.imshow(new_map)
    plt.show()

    return new_map

class TestMap:
    def __init__(self, name=name10):
        self.name = name
        self.map_dim = 1000

        self.race_map = None
        self.start = [36, 76]
        self.end = [948, 704] 

        self.x_bound = [1, 999]
        self.y_bound = [1, 999]

        self.create_race_map()

    def create_race_map(self):
        array = np.load(self.name)
        new_map = np.zeros((self.map_dim, self.map_dim))
        block_range = self.map_dim / array.shape[0]
        for i in range(self.map_dim):
            for j in range(self.map_dim):
                new_map[i, j] = array[int(i/block_range), int(j/block_range)]

        self.race_map = new_map

    def _check_location(self, x):
        if self.x_bound[0] > x[0] or x[0] > self.x_bound[1]:
            return True
        if self.y_bound[0] > x[1] or x[1] > self.y_bound[1]:
            return True 

        if self.race_map[int(x[0]), int(x[1])]:
            return True

        return False

    def show_map(self):
        plt.imshow(self.race_map)
        plt.show()


if __name__ == "__main__":
    test_map = TestMap(name00)
    test_map.show_map()