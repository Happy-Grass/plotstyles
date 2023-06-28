from matplotlib.figure import Figure
import tools
class Plotter(Figure):
    def __init__(self, figsize=(12, 8), dpi=500, **kwargs):
        super().__init__(figsize=tools.cm2inch(figsize), dpi=dpi, **kwargs)
        self.__axes = []
        self.__handles = []
        self.__legend_labels = []
        self.__x_labels = []
        self.__y_labels = []

    def draw_type(self):
        pass


    def get_axes(self):
        return self.__axes
demo = Plotter()


