from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# x = np.linspace(0, 10, num=11, endpoint=True)
# y = np.cos(-x**2/9.0)
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')

# xnew = np.linspace(0, 10, num=41, endpoint=True)

# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()

import beizer

nodes = np.array([[0, 0], [0.25, 0.4], [0.75, 0.5], [1.0, 0]])
curve1 = beizer.Curve(nodes, degree=2)

import seaborn as sns
sns.set()
ax = curve1.plot(num_pts=256)


