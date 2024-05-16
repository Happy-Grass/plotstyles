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
        # "font.family": "serif",
        "font.size": 10,
        # "mathtext.fontset": "stix",  # matplotlib渲染数学字体，和Times New Roman差别不大,也许可以删了
        # "font.serif": ["Times New Roman + SimSun"],  # Times New Roman + 宋体合并
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
# 季节负荷分配图
# 斜线比例图添加比例
def add_prop_annote(ax, color='grey', linewidth=0.5, linestyle=(0, (10, 10)), fontsize=10,
                    top_visible=True, right_visible=True, ylabel='Text'):
    """
    保证传入的xlim和ylim要相等,设置aspect为1吧,别的没有测试
    """
    min_value, max_value = ax.get_xlim()

    def get_scale(prop):
        """
        汛期占比转化为汛期与非汛期之比
        """
        return prop / (1 - prop)

    ticks_ax = ax.inset_axes([0, 0, 1, 1])
    ticks_ax.set_facecolor('None')
    ticks_ax.yaxis.tick_right()
    ticks_ax.yaxis.set_label_position("right")
    ticks_ax.xaxis.tick_top()
    ticks_ax.set_xscale('log')
    ticks_ax.set_yscale('log')
    ticks_ax.set_xlim(ax.get_xlim())
    ticks_ax.set_ylim(ax.get_ylim())
    ticks_ax.minorticks_off()

    ticks_ax.plot([min_value, max_value/get_scale(0.01)], [min_value * get_scale(0.01), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 1%
    ticks_ax.plot([min_value, max_value/get_scale(0.1)], [min_value * get_scale(0.1), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 10%
    ticks_ax.plot([min_value, max_value/get_scale(0.2)], [min_value * get_scale(0.2), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 20%
    ticks_ax.plot([min_value, max_value/get_scale(0.3)], [min_value * get_scale(0.3), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 30%
    ticks_ax.plot([min_value, max_value/get_scale(0.4)], [min_value * get_scale(0.4), max_value], linewidth=linewidth, color=color, linestyle=linestyle) # 40%
    ticks_ax.plot([min_value, max_value], [min_value, max_value], linewidth=linewidth, color='k', linestyle='-')  # 50%
    ticks_ax.plot([min_value, max_value/get_scale(0.6)], [min_value * get_scale(0.6), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 60%
    ticks_ax.plot([min_value, max_value/get_scale(0.7)], [min_value * get_scale(0.7), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 70%
    ticks_ax.plot([min_value, max_value/get_scale(0.8)], [min_value * get_scale(0.8), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 80%
    ticks_ax.plot([min_value, max_value/get_scale(0.9)], [min_value * get_scale(0.9), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 90%
    ticks_ax.plot([min_value, max_value/get_scale(0.99)], [min_value * get_scale(0.99), max_value], linewidth=linewidth, color=color, linestyle=linestyle)  # 99%
    ticks_ax.set_xticks([max_value/get_scale(0.99), max_value/get_scale(0.90), max_value/get_scale(0.80),
                        max_value/get_scale(0.70), max_value/get_scale(0.60)], ['99', '90', '', '70', '']
                        )

    ticks_ax.set_yticks([max_value * get_scale(0.01), max_value * get_scale(0.10), max_value * get_scale(0.20),
                        max_value * get_scale(0.30), max_value * get_scale(0.40), max_value * get_scale(0.50)], 
                        ['1', '10', '', '30', '', '50'])

    ticks_ax.tick_params(labelsize=fontsize)
    ticks_ax.tick_params(axis='both', which='major', length=3, pad=1)
    ticks_ax.tick_params(axis='both', which='minor', length=1)
    ticks_ax.xaxis.set_visible(top_visible)
    ticks_ax.yaxis.set_visible(right_visible)
    ticks_ax.set_ylabel(ylabel, fontsize=fontsize)
    return


def add_shade(nonflood_data, flood_data, ax, color):
    mean_x, mean_y = nonflood_data.mean(), flood_data.mean()
    ymin, ymax = ax.get_ylim()
    slopes = flood_data/nonflood_data
    max_slope, min_slope = max(slopes), min(slopes)
    mean_slope = slopes.mean()
    ax.scatter(mean_x, mean_y, marker='+', s=70, color=color)
    ax.text(mean_x, mean_y, '   {:.2f} '.format(mean_slope/(1+mean_slope) * 100), rotation=45)
    ax.fill_betweenx([ymin, ymax], [ymin / max_slope, ymax/max_slope], [ymin / min_slope, ymax/min_slope], alpha=0.2, color=color)
    return


def add_error_bar(nonflood_data, flood_data, ax, color, markersize=1, fontsize=8, fontcolor='k', linewidth=0.5, zorder=1):
    prop = flood_data/nonflood_data
    total = nonflood_data + flood_data
    prop_mean = prop.mean()
    total_mean = total.mean()
    prop_sem = prop.std()/np.sqrt(nonflood_data.size)
    total_sem = total.std()/np.sqrt(nonflood_data.size)
    # center point
    flood_p = prop_mean/(1 + prop_mean)
    nonflood_p = 1 - flood_p
    x, y = total_mean * nonflood_p, total_mean * flood_p
    ax.scatter(x, y, marker='h', s=1, color='k', zorder=zorder + 1)
    ax.text(x, y, f'     ({total_mean:.2f}t, {prop_mean/(1 + prop_mean)*100:.2f})', 
            va='top', ha='left', fontsize=fontsize, color=fontcolor)

    # total bar
    verts = [[-0.5, 0.5], [0.5, -0.5]]
    t1, t2 = total_mean - total_sem, total_mean + total_sem
    ax.plot([t1 * nonflood_p, t2 * nonflood_p], [t1 * flood_p, t2 * flood_p], 
            marker=verts, markersize=markersize, color=color, linewidth=linewidth, zorder=zorder)

    # prop bar
    verts = [[-0.5, -0.5], [0.5, 0.5]]
    p1, p2 = prop_mean  - prop_sem,   prop_mean + prop_sem
    x = []
    y = []
    for i in np.linspace(p1, p2, 100, endpoint=True):
        i1, i2 = i/(1 + i), 1/(1 + i)
        x.append(i2 * total_mean)
        y.append(i1 * total_mean)
    ax.plot(x, y, marker=verts, markersize=markersize, markevery=[0, -1], color=color, zorder=zorder, linewidth=linewidth)


# 利用四分位距剔除异常值
def eliminate_outliers(data):
    q25, q75 = np.nanquantile(data, 0.25), np.nanquantile(data, 0.75)
    iqr = q75 - q25  
  
    # 使用1.5倍四分位距找出异常值  
    lower_bound = q25 - 1.5 * iqr  
    upper_bound = q75 + 1.5 * iqr  
    mask = (data >= lower_bound) & (data <= upper_bound)  
    total_num = 1
    for s in data.shape:
        total_num = total_num * s
    print(f"The total number of outliers excluded is:{total_num - mask.sum()}")
    data[~mask] = np.nan  
    return data
