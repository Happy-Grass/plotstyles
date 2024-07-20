from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt
import plotstyles
from functools import wraps

def reload_user_styles(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # register the included stylesheet in the matplotlib style library
        plotstyles_path = plotstyles.__path__[0]
        styles_path = join(plotstyles_path, 'styles')

        # Reads styles in /styles
        stylesheets = plt.style.core.read_style_directory(styles_path)
        # Reads styles in /styles subfolders
        for inode in listdir(styles_path):
            new_data_path = join(styles_path, inode)
            if isdir(new_data_path):
                new_stylesheets = plt.style.core.read_style_directory(new_data_path)
                stylesheets.update(new_stylesheets)

        plt.style.core.reload_library()
        plt.style.core.update_nested_dict(plt.style.library, stylesheets)
        plt.style.core.available[:] = sorted(plt.style.library.keys())
        return f(*args, **kwargs)
    return decorated


@reload_user_styles
def use(style):
    plt.style.use(style)

class context:  
    def __init__(self, style):  
        self.style = style  
  
    def __enter__(self):  
        use(self.style)  
        return self  # 通常返回自己，但这里可能不需要特别返回什么  
  
    def __exit__(self, exc_type, exc_val, exc_tb):  
        use('default')  
        return False  # 这里返回 False 表示我们没有处理异常

@reload_user_styles
def available(is_print=True):
    if(is_print):
        for i in plt.style.available:
            print(i)
    return plt.style.available
