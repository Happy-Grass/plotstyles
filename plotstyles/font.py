import json
from matplotlib.font_manager import FontProperties

class FontStyle:
    def __init__(self, style='serif'):
        self.font_config_path = './plotstyles/styles/fonts/setting.json'
        self.base = FontProperties()
        self.title = FontProperties()
        self.axislabel = FontProperties()
        self.tickslabel = FontProperties()
        self.legend = FontProperties()
        self.mixed = FontProperties()
        self.mixed.set_file('./plotstyles/styles/fonts/simhei.ttf')

    def __load_config(self):
        with open(self.font_config_path, 'r') as f:
            font_config = json.load(f) 



import matplotlib.pyplot as plt

fs = FontStyle()

figure = plt.figure()
ax = figure.add_axes()
ax.text()