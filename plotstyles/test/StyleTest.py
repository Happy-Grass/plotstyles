import plotstyles.styles as style
from plotstyles.figure import Figure
import numpy as np

x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.sin(2 * x)
y3 = np.sin(3 * x)
style.available()

with style.context(['bright']):
    fig = Figure(9, 9)
    ax = fig.add_subplot(111)
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.plot(x, y3)


fig.show()