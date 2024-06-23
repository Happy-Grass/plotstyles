from matplotlib.figure import Figure as MatFigure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import tkinter as tk  
import ctypes



class Figure(MatFigure):
    def __init__(
        self,
        width=16,
        height=9,
        dpi=None,
        facecolor=None,
        edgecolor=None,
        linewidth=None,
        frameon=None,
        subplotpars=None,
        tight_layout=None,
        constrained_layout=None,
        layout=None,
        number=1,
        **kwargs
    ):
        # 尺寸转化为cm
        self.width = width * 0.3937
        self.height = height * 0.3937
        super().__init__(
            figsize=(self.width, self.height),
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            frameon=frameon,
            subplotpars=subplotpars,
            tight_layout=tight_layout,
            constrained_layout=constrained_layout,
            layout=layout,
            **kwargs
        )
        self.axes_dict = {}
        self.__axes_loc_dict = {}
        self.number = number # 用于辅助plot显示图形管理fig

    def add_axes_cm(self, name, loc_x, loc_y, width, height, precision=0.01):
        """
        添加ax,按照常用的那种cm布局, 左下角为0cm, 0cm; 右上角为 width, height
        precision: 网格的划分精度为0.01cm
        """
        nrows = int(self.height / precision)
        ncols = int(self.width / precision)
        gs = GridSpec(
            nrows=nrows,
            ncols=ncols,
            left=0,
            bottom=0,
            right=1,
            top=1,
            wspace=0,
            hspace=0,
        )
        loc_x = loc_x * 0.3937
        loc_y = loc_y * 0.3937
        width = width * 0.3937
        height = height * 0.3937

        left_bottom_x = int(loc_x/precision)
        left_bottom_y = int((self.height - loc_y)/precision)
        right_upper_x = int((width + loc_x)/precision)
        right_upper_y = int((self.height - loc_y - height)/precision)
        ax = self.add_subplot(gs[right_upper_y:left_bottom_y, left_bottom_x:right_upper_x])
        self.axes_dict[name] = ax
        self.__axes_loc_dict[name] = [loc_x, loc_y, width, height]
        return ax

    def resize(self, width, height):
        self.set_size_inches(width * 0.3937, height * 0.3937)
        return
    
    def show(self):

        # 高分屏绘制防止模糊
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        scale_factor=ctypes.windll.shcore.GetScaleFactorForDevice(0)

        window = tk.Tk()
        window.tk.call('tk', 'scaling', scale_factor/80)
        window.wm_title("Matplotlib")  


        canvas = FigureCanvasTkAgg(self, master=window)  
        toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
        canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)

        toolbar.pack(side=tk.TOP, fill=tk.X)
        widget = canvas.get_tk_widget()  
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  
        window.mainloop()

  
if __name__ == "__main__":  
    fig = Figure()
    ax = fig.add_axes_cm("Test", 2, 3, 6, 4)
    ax2 = fig.add_axes_cm("Demo", 3, 4, 3, 4)
    fig.axes_dict["Test"].plot([4, 6, 7, 8, 9])
    fig.resize(16, 8)
    fig.show()