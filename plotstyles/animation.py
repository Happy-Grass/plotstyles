from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class UpdateFrame:
    """
    This is class for FunctionAnimation in Matplotlib, Get an easy way to creat an animation
    in a specific axes, only for 2-D plot, a 3-D plot hasn't been test!
    """
    def __init__(self, ax, plot_type="line", string="Frame: {:.2f}", **kwargs):

        self.__ax = ax
        self.__plot_type = plot_type
        self.string = string
        self.__x = []
        self.__y = []

        if self.__plot_type == "line":
            line, = self.__ax.plot(self.__x, self.__y, **kwargs)
            self.__text = self.__ax.text( 0.75, 0.9, self.string.format(0), transform=self.__ax.transAxes, color='grey')
            self.__artist = line
            self.__set_data = self.__artist.set_data

        elif self.__plot_type == "scatter":
            artist = self.__ax.scatter(self.__x, self.__y, **kwargs)
            self.__text = self.__ax.text( 0.75, 0.9, self.string.format(0), transform=self.__ax.transAxes, color='grey')
            self.__artist = artist
            self.__set_data = self.__artist.set_offsets
        else:
            raise TypeError("A wrong plot type!")
        
    def __call__(self, data):
        """
        @params: data must be formatted as a tuple.
        """
        x, y = data
        self.__x.append(x)
        self.__y.append(y)
        self.__text.set_text(self.string.format(x))
        xmin = min(self.__x) - 0.2 * abs(min(self.__x))
        xmax = max(self.__x) + 0.2 * abs(max(self.__x))
        ymin = min(self.__y) - 0.2 * abs(min(self.__y))
        ymax = max(self.__y) + 0.2 * abs(max(self.__y))
        if self.__plot_type == 'scatter':
            data = np.stack([self.__x, self.__y]).T
            self.__set_data(data)
        else:
            self.__set_data(self.__x, self.__y)
        self.__ax.set_xlim(xmin, xmax)
        self.__ax.set_ylim(ymin, ymax)
        return self.__artist
    
    def get_artist_num(self):
        return len(self.__artist)

# 数据生成器
def data_gen(max_num):
    for i in range(1, max_num):
        x = i/10
        y = np.sin(x)
        yield (x, y)



if __name__ == "__main__":

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.hlines(0, 0, 50, color='b', linewidth=1, linestyles='dashed')
    ax2.hlines(0, 0, 50, color='b', linewidth=1, linestyles='dashed')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    UF1 = UpdateFrame(ax1, "line", string="X = {}", c='r', linewidth=2)
    UF2 = UpdateFrame(ax2, "scatter", string="X = {}", c='r', s=2)
    data_count = data_gen(200)
    ani1 = FuncAnimation(fig, UF1, frames=data_count, save_count=100)
    ani2 = FuncAnimation(fig, UF2, frames=data_count, save_count=100)
    # ani.save("line.mp4")
    plt.pause(1)
    plt.show()
    

