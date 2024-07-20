from plotstyles.figure import Figure
from plotstyles.fonts import global_fonts
import numpy as np

x = np.arange(0, 10, 0.1)

y0 = 2 * x + 1

y1 = 1.68 * x + 1 + np.random.normal(0, 1, len(x))
y2 = 2 * x + 1 + np.random.normal(0, 4, len(x))
y3 = 1.98 * x + 1 + np.random.normal(0, 3, len(x))

Y = np.hstack([y1.reshape(-1, 1), y2.reshape(-1, 1), y3.reshape(-1, 1)])

fig = Figure(9, 9)
ax = fig.add_subplot(111)
fig.taylor_plot(ax, x, Y, Normalize=True, scale=1, markersize=5)
ax.set_xlim(0, 8)
fig.show()