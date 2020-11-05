import numpy as np 
import glob
from matplotlib import pyplot as plt


path = 'Vehicles/TrainData/'
d_name = path + 'data*.npy'
o_name = path + 'obs*.npy'

ds = list(sorted(glob.glob(d_name)))
os = list(sorted(glob.glob(o_name)))

d_sets = []
o_sets = []


for f in ds:
    d = np.load(f)
    d_sets.append(d)
d_sets = np.array(d_sets)

for f in os:
    o = np.load(f)
    o_sets.append(o)
o_sets = np.array(o_sets)

N = len(o_sets)
print(f"Sets opened: {N}")


plt.figure(1)

start = [2, 1] 
end = [2, 23]

xstart = [start[0] , end[0]]
ystart = [start[1] , end[1]]

for i in range(N):
    plt.clf()
    plt.xlim([-0.5, 5.5])
    plt.ylim([25, 0])

    plt.plot(xstart, ystart, '--', linewidth=2)

    o_map = np.zeros((5, 25))
    for l in o_sets[i]:
        x = int(round(l[0])) 
        y = int(round(l[1]))
        o_map[x, y] = 1
    plt.imshow(o_map.T)

    xs, ys = [], []
    for pt in d_sets[i]:
        xs.append(pt[0] - 0.5) 
        ys.append(pt[1])
    
    plt.plot(xs, ys, linewidth=4)

    plt.show()



