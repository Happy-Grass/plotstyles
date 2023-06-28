import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.transforms as transforms


def circle_marker(x,
                  y,
                  r,
                  ax,
                  innerstyle=None,
                  innercolor=None,
                  edgecolor='r',
                  transform=None):
    """
    transform:默认数据坐标系，ax.transAxes来搞ax的相对坐标
    """
    #     fc_to_dc = ax.transData.inverted().transform
    #     fc_to_ndc = ax.transAxes.inverted().transform
    # r半径如果用数据轴以x轴为准fc_xr为figure坐标系下x轴方向的r像素值，另y也一样就是圆了
    dc_xr = r
    print(ax.get_ylim())
    print(ax.get_xlim())
    fc_dx, fc_dy = ax.transData.transform([x + r, y + r
                                           ]) - ax.transData.transform([x, y])
    fc_x, fc_y = ax.transData.transform([x, y])
    print(fc_dx, fc_dy)
    r_x, r_y = ax.transData.inverted().transform([
        fc_x + fc_dx, fc_y + fc_dx
    ]) - ax.transData.inverted().transform([fc_x, fc_y])
    print(r_x, r_y)

    # 校正r，x和y轴的比例如何，要保证为圆
    circle = Ellipse((x, y),
                     width=2 * r_x,
                     height=2 * r_y,
                     zorder=1,
                     color='g')
    aa = ax.add_patch(circle)
    segment = None
    lc = None
    if innerstyle == '+':
        line1 = ((x, y - r_y), (x, y + r_y))
        line2 = ((x - r_x, y), (x + r_x, y))
        segment = (line1, line2)
    if innerstyle == '-':
        line = ((x - r_x, y), (x + r_x, y))
        segment = (line, )
    if segment:
        lc = LineCollection(segment)
    if lc:
        bb = ax.add_collection(lc)
    return (aa, bb)
