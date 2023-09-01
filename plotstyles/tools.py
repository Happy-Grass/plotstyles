from matplotlib import rcParams
import numpy as np
from scipy.stats import norm
from scipy.special import gamma as gamma_fun
from scipy import integrate

def cm2inch(inch_list: list):
    """
    将单位为cm的数据转为单位为inch
    """
    cm_list = []
    for i in inch_list:
        cm_list.append(i * 0.3937)
    return cm_list


def set_mpl_rcParams(**kwargs):
    """
    设置matplotlib 的绘图基本参数
    """
    default_config = {
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset":
        "stix",  # matplotlib渲染数学字体，和Times New Roman差别不大,也许可以删了
        "font.serif": ["Times New Roman + SimSun"],  # Times New Roman + 宋体合并
        "axes.unicode_minus": False,  # 处理负号
        "axes.linewidth": 0.5,  # 坐标轴设置为0.5磅
        "xtick.major.width": 0.5,  # xticks的粗细
        "xtick.minor.width": 0.375,
        "ytick.major.width": 0.5,  # yticks的粗细
        "ytick.minor.width": 0.375,
        "lines.linewidth": 0.75,  # 画线粗细
        "lines.markersize": 3,  # 画线时点标记大小
        "grid.linestyle": "--",  # 网格线类型
        "grid.linewidth": 0.5,  # 网格线粗细
        "hatch.color": "grey",
        "hatch.linewidth": 0.2
    }

    # 将修改的参数赋值给默认参数
    for k, v in kwargs:
        default_config[k] = v
    rcParams.update(default_config)
    return


def get_color_list(name):
    color = {"default": ["#FA7F6F", "#FFBE7A", "#8ECFC9", "#82B0D2"]}
    return color[name]

# 对于临时的，可以参考这个
def set_xscale(ax, scale_name=None):
    if scale_name == None:
        print("Please assign a value to the variable scale_name!")
    if scale_name == 'Hessian':
        def forward(x):
            return norm.ppf(x) - norm.ppf(0.0001)

        def inverse(x):
            return norm.cdf(x + norm.ppf(0.00001))

        ax.set_xscale('function', functions=(forward, inverse))

    else:
        print("Error, You should assign a correct xscale_name!")


#---------------------------------------皮尔逊三型曲线相关----------------------------
def get_p3_theory_value(x_act, X_avg, C_v, C_s):
    """
    根据实际流量值求理论频率
        Parameters
        ----------
        x_act : 实际流量值, 单个值，例如:x_act = 10
        X_avg : 统计参数流量平均值
        C_v:    统计参数变差系数
        C_s:    统计参数偏态系数
        三个统计变量的调整对于理论曲线的变化需要查看相关资料，调线
        """
    # 由给定的流量计算理论频率
    a0 = X_avg * (1 - 2 * C_v / C_s)
    alpha = 4 / C_s / C_s
    beta = 2 / X_avg / C_v / C_s

    def part_p3_pline(x):
        return np.power((x - a0), (alpha - 1)) * np.exp(beta * (a0 - x))

    result = np.power(beta, alpha) / gamma_fun(alpha) * integrate.quad(part_p3_pline, x_act, np.inf)[0]
    return result

# 弦截法求方程
def solver(func, ca=0.00000000001, max_iterations=100, x0=0, x1=1):
    k = 0
    while abs(func(x1)) > ca and k < max_iterations:
        x_next = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        x0 = x1
        x1 = x_next
        k = k + 1
    return x1

# 还是有瑕疵，以后在改吧，这个积分定义域有限
def get_value4p3(p, X_avg, C_v, C_s):
    # 由给定的频率求皮尔逊三型曲线流量
    a0 = X_avg * (1 - 2 * C_v / C_s)
    def func(x):
        if x <= a0:
            result = -p
        else:
            result = get_p3_theory_value(x, X_avg, C_v, C_s) - p
        return result
    # 求解gamma函数的积分解需要注意流量初值，x > a0

    return solver(func=func, x0=0, x1=a0 + 10)


#-----------------------------------------------------------------------------------------------
