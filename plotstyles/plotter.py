import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import functools
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.mlab as mlab


def adjacent_values(vals, q1, q3):
    # vals 一组原始数据
    # q1 四分之一
    # q3 四分之三
    # 返回离群上限和离群下限(超过了最大值就修正为最大值)
    vals = sorted(vals)
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def violin_plot(data,
                ax,
                colors=[],
                labels=[],
                linestyles=None,
                boldlw=1,
                lightlw=0.3,
                markersize=0.5):
    """
    @description  :
    plot a violingraph
    ---------
    @param  :
    Data: a list of different categrory data, such as
    Data = [np.array, np.array, np.array]
    ax: a axes
    -------
    @Returns  : the plot object
    -------
    """
    for i, d in enumerate(data):
        m, = d.shape
        data[i] = d[~np.isnan(d)]
        n, = data[i].shape
        if m != n:
            print('请注意：第{}组数据剔除了空值！'.format(i + 1))
    if len(labels) == 0:
        labels = range(1, len(data) + 1)
    if len(colors) == 0:
        colors = len(data) * ["#D43F3A"]
    parts = ax.violinplot(dataset=data, showextrema=False, widths=0.50)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor("k")
        pc.set_linewidth(0.1)
        pc.set_alpha(1)
    if linestyles is not None:
        for pc, ls in zip(parts["bodies"], linestyles):
            pc.set_linestyle(ls)

    # 获取四分之一位距，四分之三位矩，中位数,并分别存入quartile1, medians, quartiles3向量
    loc = np.percentile(data[0], [25, 50, 75])
    for i in data[1::]:
        temp_loc = np.percentile(i, [25, 50, 75])
        loc = np.c_[loc, temp_loc]

    quartile1, medians, quartile3 = loc[0, :], loc[1, :], loc[2, :]

    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
    ])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)  # 分段数
    # 中位数标记
    ax.scatter(inds,
               medians,
               marker="o",
               color="white",
               s=markersize,
               zorder=3)
    # 四分位粗线
    ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=boldlw)
    ax.vlines(inds,
              whiskers_min,
              whiskers_max,
              color="k",
              linestyle="-",
              lw=lightlw)
    # 离群细线
    # set style for the axes
    ax.set_xticks(np.arange(1, len(medians) + 1))
    ax.set_xlim(0.25, len(medians) + 0.75)
    ax.set_xticklabels(labels=labels)


def errorbar(Data, ax, x=None, **kwargs):
    for i, d in enumerate(Data):
        m, = d.shape
        Data[i] = d[~np.isnan(d)]
        n, = Data[i].shape
        if m != n:
            print('请注意：第{}组数据剔除了空值！'.format(i + 1))

    if x is None:
        x = range(len(Data))
    ave = [np.mean(data) for data in Data]
    sigma = [np.std(data) for data in Data]
    length = [data.shape[0] for data in Data]
    sem = []
    for sig, length in zip(sigma, length):
        sem.append(sig / np.sqrt(length))
    patch = ax.bar(x=x, height=ave, yerr=[[0] * len(sem), sem], **kwargs)

    return patch


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# 分两边的小提琴图，用于不同季节对比
def violin(ax, vpstats, positions=None, offset=0, vert=True, widths=0.5, half='all',
               showmeans=False, showextrema=False, showmedians=False, 
               fillcolor=None, edgecolor=None, linecolor=None, pdata=None, showquantiles=True):
        """
        绘制小提琴图的函数.
        为每一列 *vpstats* 绘制小提琴图，每一个填充的面积用于代表整个数据范围，可以选择平均值，中位数，最小值，最大值和其它分位数

        参数
        ----------
        vpstats : 字典列表，应当包含一下统计参数 
          应当包括以下键值对:
          - ``coords``: 标量列表包含高斯核密度估计的坐标值

          - ``vals``: A list of scalars containing the values of the
            kernel density estimate at each of the coordinates given
            in *coords*.
          - ``mean``: 小提琴图的平均值.
          - ``median``: 小提琴图的中位数.
          - ``min``:  小提琴图的最小值.
          - ``max``: 小提琴图的最大值.
          可选的键值对:
          - ``quantiles``: 小提琴图的分位值列表

        positions : 位置列表，默认为: [1, 2, ..., n]
        vert : bool, default: True, 为True则绘制垂直，否则绘制竖直.
        widths : array-like, default: 0.5, 标量或者向量用于设置violin的最大宽度
        showmeans : bool, default: False
        showextrema : bool, default: True
        showmedians : bool, default: False
        Returns
        -------
        dict
            返回字典，字典中存储violinplot的每个组分。
          - ``bodies``: A list of the `~.collections.PolyCollection`
            instances containing the filled area of each violin.
          - ``cmeans``: A `~.collections.LineCollection` instance that marks
            the mean values of each of the violin's distribution.
          - ``cmins``: A `~.collections.LineCollection` instance that marks
            the bottom of each violin's distribution.
          - ``cmaxes``: A `~.collections.LineCollection` instance that marks
            the top of each violin's distribution.
          - ``cbars``: A `~.collections.LineCollection` instance that marks
            the centers of each violin's distribution.
          - ``cmedians``: A `~.collections.LineCollection` instance that
            marks the median values of each of the violin's distribution.
          - ``cquantiles``: A `~.collections.LineCollection` instance created
            to identify the quantiles values of each of the violin's
            distribution.
        """
        # 小提琴图的组件
        means = [] # 均值线
        mins = [] # 最小值线
        maxes = [] # 最大值线
        medians = [] # 中位线
        quantiles = [] # 分位线
        qlens = []  # 每一个数据集有的分位线个数.
        artists = {}  # 返回的字典

        N = len(vpstats)
        datashape_message = ("List of violinplot statistics and `{0}` "
                             "values must have the same length")

        # Validate positions
        if positions is None:
            positions = range(1, N + 1)
        elif len(positions) != N:
            raise ValueError(datashape_message.format("positions"))

        # Validate widths
        if np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format("widths"))

        # Colors.
        if fillcolor == None:
            fillcolor = 'r'
        if linecolor == None:
            linecolor = 'b'
        if edgecolor == None:
            edgecolor = 'k'

        # Check whether we are rendering vertically or horizontally
        if vert:
            fill = ax.fill_betweenx
            perp_lines = functools.partial(ax.hlines, colors=linecolor)
            par_lines = functools.partial(ax.vlines, colors=linecolor)
        else:
            fill = ax.fill_between
            perp_lines = functools.partial(ax.vlines, colors=linecolor)
            par_lines = functools.partial(ax.hlines, colors=linecolor)

        # Render violins
        bodies = []
        ttt = []
        for stats, pos, width in zip(vpstats, positions, widths):
            # The 0.5 factor reflects the fact that we plot from v-p to v+p.
            vals = np.array(stats['vals'])
            ttt.append(vals.max())
            vals = 0.5 * width * vals / vals.max()
            if half == 'all':
                bodies += [fill(stats['coords'], -vals + pos, vals + pos,
                              facecolor=fillcolor, alpha=0.3, edgecolor='k', linewidth=0.4)]
                if len(ttt) == len(positions):
                    ttt = np.array(ttt)
                    line_ends = {}
                    for k, v in pdata.items():
                        line_ends[k] = [-np.array(v) * 0.5 * width/ttt + positions, 
                                        np.array(v) * 0.5 * width/ttt + positions]
                # line_ends = [[-0.25], [0.25]] * np.array(widths) + positions
            if half == 'left':
                bodies += [fill(stats['coords'], -vals + pos - offset, np.zeros_like(vals) + pos - offset,
                              facecolor=fillcolor, alpha=0.3, edgecolor='k', linewidth=0.1)]
                # line_ends = [[-0.25], [0]] * np.array(widths) + positions
                if len(ttt) == len(positions):
                    ttt = np.array(ttt)
                    line_ends = {}
                    for k, v in pdata.items():
                        line_ends[k] = [-np.array(v) * 0.5 * width/ttt + positions - offset, 
                                        np.array(positions) - offset]
            if half == 'right':
                bodies += [fill(stats['coords'], np.zeros_like(vals) + pos + offset, vals + pos + offset,
                              facecolor=fillcolor, alpha=0.3, edgecolor='k', linewidth=0.1)]
                # Calculate ranges for statistics lines (shape (2, N)).
                # line_ends = [[0], [0.25]] * np.array(widths) + positions
                if len(ttt) == len(positions):
                    ttt = np.array(ttt)
                    line_ends = {}
                    for k, v in pdata.items():
                        line_ends[k] = [np.array(positions) + offset, 
                                        np.array(v) * 0.5 * width/ttt + positions + offset]
            means.append(stats['mean'])
            mins.append(stats['min'])
            maxes.append(stats['max'])
            medians.append(stats['median'])
            q = stats.get('quantiles')  # a list of floats, or None
            if q is None:
                q = []
            quantiles.extend(q)
            qlens.append(len(q))
        artists['bodies'] = bodies

        if showmeans:  # Render means
            artists['cmeans'] = perp_lines(means, *line_ends['mean'])
        if showextrema:  # Render extrema
            artists['cmaxes'] = perp_lines(maxes, *line_ends)
            artists['cmins'] = perp_lines(mins, *line_ends)
            artists['cbars'] = par_lines(positions, mins, maxes)
        if showmedians:  # Render medians
            artists['cmedians'] = perp_lines(medians, *line_ends['median'], linewidth=0.8, color=fillcolor, alpha=0.5)
        if showquantiles:  # Render quantiles: each width is repeated qlen times.
            artists['q14'] = perp_lines(pdata['q1'], *line_ends['quan1'], linewidth=0.4, color=fillcolor, linestyles='dashed', alpha=0.5)
            artists['q34'] = perp_lines(pdata['q3'], *line_ends['quan3'], linewidth=0.4, color=fillcolor, linestyles='dashed', alpha=0.5)

        return artists

def violinplot(ax, dataset, positions=None, offset=0, vert=True, widths=0.5, half='all',
                   showmeans=False, showextrema=False, showmedians=False,
                   quantiles=None, points=300, bw_method=None, linecolor=None, fillcolor=None, edgecolor=None, showquantiles=True):
        def _kde_method(X, coords):
            # Unpack in case of e.g. Pandas or xarray object
            X = cbook._unpack_to_numpy(X)
            # fallback gracefully if the vector contains only one value
            if np.all(X[0] == X):
                return (X[0] == coords).astype(float)
            kde = mlab.GaussianKDE(X, bw_method)
            return kde.evaluate(coords)
        # dataset里面最好都是numpy ba，这里主要是确定mean，median， 1/4， 3/4的深度
        quan1_4p, medianp, quan3_4p, mean_valuep = [], [], [], []
        quans1_4, quans3_4 = [], []
        for data in dataset:
            quan1_4, median, quan3_4 = np.percentile(data, [25, 50, 75])
            quans1_4.append(quan1_4)
            quans3_4.append(quan3_4)
            mean_value = data.mean()
            quan1_4p.append(_kde_method(data, quan1_4)[0])
            quan3_4p.append(_kde_method(data, quan3_4)[0])
            medianp.append(_kde_method(data, median)[0])
            mean_valuep.append(_kde_method(data, mean_value)[0])
        p_data = {"q1": quans1_4, "q3": quans3_4, "quan1": quan1_4p, "quan3": quan3_4p, "median": medianp, "mean": mean_valuep}

        vpstats = cbook.violin_stats(dataset, _kde_method, points=points,
                                     quantiles=quantiles)
        return violin(ax, vpstats, positions=positions, offset=offset, vert=vert, half=half,
                           widths=widths, showmeans=showmeans, fillcolor=fillcolor, pdata=p_data, showquantiles=showquantiles,
                           showextrema=showextrema, showmedians=showmedians, linecolor=linecolor, edgecolor=edgecolor)

def distribution_line(ax, dataset, positions=None, center_linewidth=1.5, linewidth=0.5):
    quans1_4 = []
    quans3_4 = []
    medians = []
    maxs = []
    mins = []
    N = len(dataset)
    if positions is None:
        positions = range(1, N + 1)
    elif len(positions) != N:
        raise ValueError("Dimension error!")
    for data in dataset:
        quan1_4, median, quan3_4 = np.percentile(data, [25, 50, 75])
        quans1_4.append(quan1_4)
        quans3_4.append(quan3_4)
        medians.append(median)
        maxs.append(max(data))
        mins.append(min(data))
    ax.vlines(positions, mins, maxs, linewidth=linewidth, color='k')
    ax.vlines(positions, quans1_4, quans3_4, linewidth=center_linewidth, color='k')
    ax.scatter(positions, medians, marker='o', s=1.5, color='white', zorder=8)

def plot_violin_binary(ax, data_left, data_right, data_all, positions=None, offset=0, leftcolor='r', rightcolor='b',linewidth=0.5, center_linewidth=1.5):
    violinplot(ax=ax, dataset=data_left, positions=positions, offset=offset, half='left', showmedians=True, showquantiles=True, fillcolor=leftcolor)
    violinplot(ax=ax, dataset=data_right, positions=positions, offset=offset, half='right', showmedians=True, showquantiles=True, fillcolor=rightcolor)
    distribution_line(ax, data_all, positions=positions, linewidth=linewidth, center_linewidth=center_linewidth)
    return None