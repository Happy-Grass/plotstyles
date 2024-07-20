import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import math


class TaylorDiagram:
    """
    reference: np.array nx1
    simulations: np.array nxm, representing m sets of simulations
    """

    def __init__(
        self,
        ax,
        reference,
        simulations,
        Normalize=False,
        markers=[],
        colors=[],
        scale=1.2,
        markersize=2,
        pkwargs={},
        reference_name='Observation',
        simulations_name=None,
        legend=False,
        r_linewidth = 1,
        r_linecolor = 'k',
        r_linestyle = '--',
        ref_std_linewidth = 1.5,
        ref_std_linecolor = 'k',
        ref_std_linestyle = '-',
        rmse_linewidth = 1,
        rmse_linecolor = 'grey',
        rmse_linestyle = '--',
    ):
        self.points = []
        self.Normalize = Normalize
        self.pkwargs = pkwargs
        self.markers = (
            markers if len(markers) else ["o", "o", "s", "v", "o", "s", "v"] * 100
        )
        self.colors = (
            colors
            if len(colors)
            else [
                "tab:blue",
                "tab:red",
                "tab:red",
                "tab:red",
                "tab:green",
                "tab:green",
                "tab:green",
                "#1abc9c",
                "#2ecc71",
                "#3498db",
                "#9b59b6",
                "#34495e",
            ]
        )
        self.markersize = markersize
        self.reference = reference
        self.reference_name = reference_name
        self.scale = scale
        self.simulations = simulations
        self.simulations_name = [str(i) for i in range(simulations.shape[1])] if simulations_name is None else simulations_name
        self.r_linewidth = r_linewidth
        self.r_linecolor = r_linecolor
        self.r_linestyle = r_linestyle
        self.ref_std_linewidth = ref_std_linewidth
        self.ref_std_linecolor = ref_std_linecolor
        self.ref_std_linestyle = ref_std_linestyle
        self.rmse_linewidth = rmse_linewidth
        self.rmse_linestyle = rmse_linestyle
        self.rmse_linecolor = rmse_linecolor

        self.step_up(ax)  # set up a diagram axes
        self.plot_sample()  # draw sample points
        if legend:
            self.legend()  # add legend

    def calc_loc(self, reference, simulation):
        R = np.corrcoef(reference, simulation)[0, 1]
        theta = np.arccos(R)
        simu_std = simulation.std()
        return theta, simu_std / self._refstd if self.Normalize else simu_std

    def step_up(self, ax):
        # close the original axis
        ax.axis("off")
        ll, bb, ww, hh = ax.get_position().bounds
        # polar transform
        tr = PolarAxes.PolarTransform()
        # theta range
        Rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        Tlocs = np.arccos(Rlocs)  # convrt to theta locations
        # grid finder
        gl1 = GF.FixedLocator(Tlocs)  # theta locator
        tf1 = GF.DictFormatter(dict(zip(Tlocs, map(str, Rlocs))))  # theta formatter

        # std range
        self._refstd = self.reference.std()
        self.stdmax = max(
            [self.simulations[:, i].std() for i in range(self.simulations.shape[1])]
            + [self._refstd]
        )
        self.Smax = (math.ceil(self.stdmax / self._refstd) if self.Normalize else math.ceil(self.stdmax)) * self.scale
        self.refstd = 1 if self.Normalize else self._refstd
        Slocs = np.linspace(0, self.Smax, 4)

        gl2 = GF.FixedLocator(Slocs)  # theta locator
        tf2 = GF.DictFormatter(
            dict(zip(Slocs, map(lambda i: "%.1f" % i, Slocs)))
        )  # theta formatter
        # construct grid helper
        grid_helper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, 0, self.Smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
            grid_locator2=gl2,
            tick_formatter2=tf2,
        )
        ax = ax.figure.add_axes(
            [ll, bb, ww, hh],
            facecolor="none",
            axes_class=FA.FloatingAxes,
            grid_helper=grid_helper,
        )
        # theta
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation Coefficient")
        ax.axis["top"].major_ticklabels.set_pad(8)

        # std bottom
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].toggle(ticklabels=False)

        # std left
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["right"].label.set_text("Standard deviation")
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].major_ticklabels.set_pad(8)
        # hide
        ax.axis["bottom"].set_visible(False)
        # draw grid
        ax.grid(linestyle=self.r_linestyle, color=self.r_linecolor, linewidth=self.r_linewidth)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Ref_STD线
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, color=self.ref_std_linecolor, linewidth=self.ref_std_linewidth, linestyle=self.ref_std_linestyle)

        # RMS格网
        rs, ts = np.meshgrid(np.linspace(0, self.Smax, 100), np.linspace(0, np.pi / 2, 100))
        rms = (self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts)) ** 0.5
        contours = self.ax.contour(
            ts,
            rs,
            rms,
            levels=np.linspace(0, self.Smax, 4) if self.Normalize else 4,
            colors=self.rmse_linecolor,
            linestyles=self.rmse_linestyle,
            linewidths=self.rmse_linewidth
        )
        self.ax.clabel(contours, contours.levels, inline=True, fmt="%.1f")
        # 绘制参考点
        p, = self.ax.plot(
            0,
            self.refstd,
            linestyle="",
            marker=self.markers[0],
            color=self.colors[0],
            markersize=self.markersize,
            **self.pkwargs
        )
        p.set_label(self.reference_name)
        p.set_clip_on(False)  # reference点不被裁剪
        self.points.append(p)

    def plot_sample(self):
        stds = []
        for i, marker, color in zip(range(self.simulations.shape[1]), self.markers[1:], self.colors[1:]):
            t, s = self.calc_loc(self.reference, self.simulations[:,i])
            p, = self.ax.plot(
                t,
                s,
                linestyle="",
                marker=marker,
                color=color,
                markersize=self.markersize,
                **self.pkwargs
            )
            p.set_label(self.simulations_name[i])
            self.points.append(p)
            stds.append(s)
        self.ax.set_xlim(xmax=max(stds))

    def legend(self):
        ll, bb, ww, hh = self.ax.get_position().bounds
        self.ax.legend(
            ncol=self.simulations.shape[1] + 1,
            loc="lower center",
            frameon=False,
            bbox_to_anchor=(ll, bb - hh * 0.3, ww, hh * 0.1),
        )


if __name__ == "__main__":
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}) 
    # ax.plot(0.5, 1)
    # plt.show()
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 7, 6], "c": [2, 3, 4, 6, 9]})
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    td = TaylorDiagram(axes, df.iloc[:, 0].values, df.iloc[:, 1:].values, markersize=5, Normalize=True, scale=2, legend=True)
    plt.show()
