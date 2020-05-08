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

x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)

from scipy.interpolate import make_lsq_spline, BSpline
t = [-1, 0, 1]
k = 3
t = np.r_[(x[0],)*(k+1),
          t,
          (x[-1],)*(k+1)]
t = np.r_[(x[0],)*(k+1),
          t,
          (x[-1],)*(k+1)]
print(t)
spl = make_lsq_spline(x, y, t, k)

from scipy.interpolate import make_interp_spline
spl_i = make_interp_spline(x, y)

import matplotlib.pyplot as plt
xs = np.linspace(-3, 3, 100)
plt.plot(x, y, 'ro', ms=5)
plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
plt.legend(loc='best')
plt.show()


