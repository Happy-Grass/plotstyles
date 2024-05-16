from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from typing import Literal
import os
from matplotlib import rcParams

class Fonts():
    zh = ['SimSun']
    en = ['Times New Roman']
    zhen_mixed = ['Times New Roman + SimSun']
    def __init__(self):
        global_font_manager = font_manager.fontManager

        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        fonts_basedir = os.path.join(current_dir, "sources", "fonts")

        # 添加包自带的字体
        font_dirs = [os.path.join(fonts_basedir, subdir) for subdir in ["ttf", "afm", "pdfcorefonts"]]
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            global_font_manager.addfont(font_file)

        
        # 默认字体配置
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Times New Roman + SimSun"]
        # 配置数学环境字体为Computer Modern Text
        rcParams["mathtext.fontset"] = 'cm'
    
    def serif(self, size: int=10, 
                 style: Literal['normal', 'italic', 'oblique'] ='normal', 
                 weight: Literal['ultralight', 'light', 'normal', 'regular', 'book', 
                                 'medium', 'roman', 'semibold', 'demibold', 'demi', 
                                 'bold', 'heavy', 'extra bold', 'black'] = 'normal',
                 variant: Literal['normal', 'small-caps'] = 'normal',
                 stretch: Literal['ultra-condensed', 'extra-condensed', 'condensed', 
                                  'semi-condensed', 'normal', 'semi-expanded', 'expanded',
                                  'extra-expanded', 'ultra-expanded'] = 'normal',
                 math_fontfamily: Literal['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans'] = 'cm'):
        return FontProperties(family=['Times New Roman', 'SimSun'], size=size, style=style, weight=weight, variant=variant,
                              stretch=stretch, math_fontfamily=math_fontfamily )
    
    def sans(self, size: int=10, 
                 style: Literal['normal', 'italic', 'oblique'] ='normal', 
                 weight: Literal['ultralight', 'light', 'normal', 'regular', 'book', 
                                 'medium', 'roman', 'semibold', 'demibold', 'demi', 
                                 'bold', 'heavy', 'extra bold', 'black'] = 'normal',
                 variant: Literal['normal', 'small-caps'] = 'normal',
                 stretch: Literal['ultra-condensed', 'extra-condensed', 'condensed', 
                                  'semi-condensed', 'normal', 'semi-expanded', 'expanded',
                                  'extra-expanded', 'ultra-expanded'] = 'normal',
                 math_fontfamily: Literal['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans'] = 'stixsans'):
        return FontProperties(family=['Arial', 'SimHei'], size=size, style=style, weight=weight, variant=variant,
                              stretch=stretch, math_fontfamily=math_fontfamily )

global_fonts = Fonts()
