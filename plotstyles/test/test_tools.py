from plotstyles import tools
import numpy as np
import matplotlib.pyplot as plt



x = np.array([0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]) * 0.01
y = np.array([500, 400, 300, 200, 150, 100, 90, 80, 70, 60, 59, 58, 57, 56, 50])
fig = plt.figure()
ax = fig.add_subplot(111)
tools.set_xscale(ax, scale_name='Hessian')
ax.set_xticks(x)

ax.plot(x, y)
plt.show()
