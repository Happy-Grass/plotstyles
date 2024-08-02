from matplotlib.axes import Axes
from matplotlib.figure import Figure as MatFigure
from typing import Iterable, Literal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import tkinter as tk
import ctypes
from matplotlib.widgets import Slider
import corner
from plotstyles.visualization.taylor.taylor_diagram import TaylorDiagram
from matplotlib.patches import ConnectionPatch
import math


class MultiSlider:
    def __init__(self, ax, func, variables: dict = {"k": [0, 1, 0.1]}, **kwargs):
        self.func = func
        self.items = {}
        subax_height = 1 / (len(variables) * 1.5 - 0.5)
        for i, (key, value) in enumerate(variables.items()):
            subax = ax.inset_axes([0.1, 1.5 * i * subax_height, 0.8, subax_height])
            subax.xaxis.set_visible(False)
            subax.yaxis.set_visible(False)
            for spine in ["top", "right", "bottom", "left"]:
                subax.spines[spine].set_visible(False)
            valmin, valmax, valstep = value
            slider = Slider(subax, key, valmin=valmin, valmax=valmax, valstep=valstep)
            self.items[key] = slider
            slider.on_changed(self.__silder_func)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def __silder_func(self, value):
        # value 对于单个button没有用,调用综合button
        args = []
        for value in self.items.values():
            args.append(value.val)
        self.func(*args)


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
        **kwargs,
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
            **kwargs,
        )
        self.axes_dict = {}
        self.number = number  # 用于辅助plot显示图形管理fig

    def add_axes_cm(
        self,
        name,
        loc_x,
        loc_y,
        width,
        height,
        anchor: Literal[
            "left bottom", "left upper", "right bottom", "right upper"
        ] = "left bottom",
        **kwargs
    ):
        """
        添加ax,按照常用的那种cm布局, 左下角为0cm, 0cm; 右上角为 width, height
        loc_x, loc_y随着anchor的变化而变化，如果是left bottom就是左下角，如果是right upper就是右上角
        """
        loc_x, loc_y, width, height = (
            loc_x * 0.3937,
            loc_y * 0.3937,
            width * 0.3937,
            height * 0.3937,
        )
        width = width / self.width
        height = height / self.height
        if anchor == "left bottom":
            left = loc_x / self.width
            bottom = loc_y / self.height
        elif anchor == "left upper":
            left = loc_x / self.width
            bottom = 1 - loc_y / self.height - height
        elif anchor == "right bottom":
            left = 1 - loc_x / self.width - width
            bottom = loc_y / self.height
        elif anchor == "right upper":
            left = 1 - loc_x / self.width - width
            bottom = 1 - loc_y / self.height - height
        elif (
            isinstance(anchor, tuple)
            and len(anchor) == 2
            and all([isinstance(item, float) for item in anchor])
        ):
            start_x, start_y = anchor
            left = start_x + loc_x / self.width
            bottom = start_y + loc_y / self.height
        else:
            raise ValueError
        ax = self.add_axes([left, bottom, width, height], **kwargs)
        self.axes_dict[name] = ax
        return ax

    def resize(self, width, height):
        self.set_size_inches(width * 0.3937, height * 0.3937)
        return

    def show(self):

        # 高分屏绘制防止模糊
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0)

        window = tk.Tk()
        window.tk.call("tk", "scaling", scale_factor / 80)
        window.wm_title("Matplotlib")

        canvas = FigureCanvasTkAgg(self, master=window)
        toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
        canvas.mpl_connect(
            "key_press_event", lambda event: print(f"you pressed {event.key}")
        )
        canvas.mpl_connect("key_press_event", key_press_handler)

        toolbar.pack(side=tk.TOP, fill=tk.X)
        widget = canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        window.mainloop()

    def align_ylabels_coords(self, axes, x, y):
        for ax in axes:
            ax.yaxis.set_label_coords(x, y, transform=ax.transAxes)

    def align_xlabels_coords(self, axes, x, y):
        """
        Align xlabels coords
        """
        for ax in axes:
            ax.xaxis.set_label_coords(x, y, transform=ax.transAxes)
        return

    def plot_line_between_2ax(
        self,
        xy1: tuple,
        xy2: tuple,
        ax1=None,
        ax2=None,
        coords1="data",
        coords2="data",
        color="k",
        linestyle="solid",
        linewidth=1,
        **kwargs
    ):
        """*coordsA* and *coordsB* are strings that indicate the
        coordinates of *xyA* and *xyB*.

        ==================== ==================================================
        Property             Description
        ==================== ==================================================
        'figure points'      points from the lower left corner of the figure
        'figure pixels'      pixels from the lower left corner of the figure
        'figure fraction'    0, 0 is lower left of figure and 1, 1 is upper
                            right
        'subfigure points'   points from the lower left corner of the subfigure
        'subfigure pixels'   pixels from the lower left corner of the subfigure
        'subfigure fraction' fraction of the subfigure, 0, 0 is lower left.
        'axes points'        points from lower left corner of the Axes
        'axes pixels'        pixels from lower left corner of the Axes
        'axes fraction'      0, 0 is lower left of Axes and 1, 1 is upper right
        'data'               use the coordinate system of the object being
                            annotated (default)
        'offset points'      offset (in points) from the *xy* value
        'polar'              you can specify *theta*, *r* for the annotation,
                            even in cartesian plots.  Note that if you are
                            using a polar Axes, you do not need to specify
                            polar for the coordinate system since that is the
                            native "data" coordinate system.
        ==================== ==================================================
        """
        con = ConnectionPatch(
            xy1,
            xy2,
            coordsA=coords1,
            coordsB=coords2,
            axesA=ax1,
            axesB=ax2,
            color=color,
            ls=linestyle,
            linewidth=linewidth,
            **kwargs
        )
        self.add_artist(con)
        return

    
    def line_annoate_text(self, ax, xy1: tuple, xy2: tuple, text: str, transform=None):
        if transform is None:
            transform = ax.transData
            coords = 'data'
        if transform == ax.transAxes:
            coords = 'axes fraction'
        theta = math.degrees(math.atan((xy2[1] - xy1[1]) / (xy2[0] - xy1[0])))
        mid_x = (xy1[0] + xy2[0]) / 2
        mid_y = (xy1[1] + xy2[1]) / 2
        ax.annotate(
            "",
            xy=xy1,
            xytext=xy2,
            arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, facecolor="k", linewidth=0.5),
            xycoords=transform,
            textcoords=transform,
        )
        ax.text(
            mid_x,
            mid_y,
            text,
            ha="center",
            va="center",
            rotation=theta,
            rotation_mode="anchor",
            transform_rotates_text=True,
            bbox=dict(
                boxstyle="square",
                ec=(1, 1, 1),
                fc=(1, 1, 1),
            ),
            transform=transform
        )
        return None

    def corner_plot(
        self,
        data,
        bins=20,
        *,
        range=None,
        axes_scale="linear",
        weights=None,
        color=None,
        hist_bin_factor=1,
        smooth=None,
        smooth1d=None,
        labels=None,
        label_kwargs=None,
        titles=None,
        show_titles=False,
        title_quantiles=None,
        title_fmt=".2f",
        title_kwargs=None,
        truths=None,
        truth_color="#4682b4",
        scale_hist=False,
        quantiles=None,
        verbose=False,
        fig=None,
        max_n_ticks=5,
        top_ticks=False,
        use_math_text=False,
        reverse=False,
        labelpad=0.0,
        hist_kwargs=None,
        # Arviz parameters
        group="posterior",
        var_names=None,
        filter_vars=None,
        coords=None,
        divergences=False,
        divergences_kwargs=None,
        labeller=None,
        **hist2d_kwargs,
    ):
        """
        Make a *sick* corner plot showing the projections of a data set in a
        multi-dimensional space. kwargs are passed to hist2d() or used for
        `matplotlib` styling.

        Parameters
        ----------
        data : obj
            Any object that can be converted to an ``arviz.InferenceData`` object.
            Refer to documentation of ``arviz.convert_to_dataset`` for details.

        bins : int or array_like[ndim,]
            The number of bins to use in histograms, either as a fixed value for
            all dimensions, or as a list of integers for each dimension.

        group : str
            Specifies which InferenceData group should be plotted.  Defaults to
            ``'posterior'``.

        var_names : list
            Variables to be plotted, if ``None`` all variable are plotted. Prefix
            the variables by `~` when you want to exclude them from the plot.

        filter_vars : {``None``, ``"like"``, ``"regex"``}
            If ``None`` (default), interpret ``var_names`` as the real variables
            names. If ``"like"``, interpret ``var_names`` as substrings of the real
            variables names. If ``"regex"``, interpret ``var_names`` as regular
            expressions on the real variables names. A la ``pandas.filter``.

        coords : mapping
            Coordinates of ``var_names`` to be plotted. Passed to
            ``arviz.Dataset.sel``.

        divergences : bool
            If ``True`` divergences will be plotted in a different color, only if
            ``group`` is either ``'prior'`` or ``'posterior'``.

        divergences_kwargs : dict
            Any extra keyword arguments to send to the ``overplot_points`` when
            plotting the divergences.

        labeller : arviz.Labeller
            Class providing the method ``make_label_vert`` to generate the labels
            in the plot. Read the ArviZ label guide for more details and usage
            examples.

        weights : array_like[nsamples,]
            The weight of each sample. If `None` (default), samples are given
            equal weight.

        color : str
            A ``matplotlib`` style color for all histograms.

        hist_bin_factor : float or array_like[ndim,]
            This is a factor (or list of factors, one for each dimension) that
            will multiply the bin specifications when making the 1-D histograms.
            This is generally used to increase the number of bins in the 1-D plots
            to provide more resolution.

        smooth, smooth1d : float
        The standard deviation for Gaussian kernel passed to
        `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
        respectively. If `None` (default), no smoothing is applied.

        labels : iterable (ndim,)
            A list of names for the dimensions.

            .. deprecated:: 2.2.1
                If a ``xs`` is a ``pandas.DataFrame`` *and* ArviZ is installed,
                labels will default to column names.
                This behavior will be removed in version 3;
                either use ArviZ data structures instead or pass
                ``labels=dataframe.columns`` manually.

        label_kwargs : dict
            Any extra keyword arguments to send to the `set_xlabel` and
            `set_ylabel` methods. Note that passing the `labelpad` keyword
            in this dictionary will not have the desired effect. Use the
            `labelpad` keyword in this function instead.

        titles : iterable (ndim,)
            A list of titles for the dimensions. If `None` (default),
            uses labels as titles.

        show_titles : bool
            Displays a title above each 1-D histogram showing the 0.5 quantile
            with the upper and lower errors supplied by the quantiles argument.

        title_quantiles : iterable
            A list of 3 fractional quantiles to show as the the upper and lower
            errors. If `None` (default), inherit the values from quantiles, unless
            quantiles is `None`, in which case it defaults to [0.16,0.5,0.84]

        title_fmt : string
            The format string for the quantiles given in titles. If you explicitly
            set ``show_titles=True`` and ``title_fmt=None``, the labels will be
            shown as the titles. (default: ``.2f``)

        title_kwargs : dict
            Any extra keyword arguments to send to the `set_title` command.

        range : iterable (ndim,)
            A list where each element is either a length 2 tuple containing
            lower and upper bounds or a float in range (0., 1.)
            giving the fraction of samples to include in bounds, e.g.,
            [(0.,10.), (1.,5), 0.999, etc.].
            If a fraction, the bounds are chosen to be equal-tailed.

        axes_scale : str or iterable (ndim,)
            Scale (``"linear"``, ``"log"``) to use for each data dimension. If only
            one scale is specified, use that for all dimensions.

        truths : iterable (ndim,)
            A list of reference values to indicate on the plots.  Individual
            values can be omitted by using ``None``.

        truth_color : str
            A ``matplotlib`` style color for the ``truths`` makers.

        scale_hist : bool
            Should the 1-D histograms be scaled in such a way that the zero line
            is visible?

        quantiles : iterable
            A list of fractional quantiles to show on the 1-D histograms as
            vertical dashed lines.

        verbose : bool
            If true, print the values of the computed quantiles.

        plot_contours : bool
            Draw contours for dense regions of the plot.

        use_math_text : bool
            If true, then axis tick labels for very large or small exponents will
            be displayed as powers of 10 rather than using `e`.

        reverse : bool
            If true, plot the corner plot starting in the upper-right corner
            instead of the usual bottom-left corner

        labelpad : float
            Padding between the axis and the x- and y-labels in units of the
            fraction of the axis from the lower left

        max_n_ticks: int
            Maximum number of ticks to try to use

        top_ticks : bool
            If true, label the top ticks of each axis

        fig : `~matplotlib.figure.Figure`
            Overplot onto the provided figure object, which must either have no
            axes yet, or ``ndim * ndim`` axes already present.  If not set, the
            plot will be drawn on a newly created figure.

        hist_kwargs : dict
            Any extra keyword arguments to send to the 1-D histogram plots.

        **hist2d_kwargs
            Any remaining keyword arguments are sent to :func:`corner.hist2d` to
            generate the 2-D histogram plots.
        Returns
        """
        if len(self.axes_dict) != 0:
            raise RuntimeError("Corner_plot should be used in a clean figure")
        if fig is None:
            fig = self
        corner.corner(
            data=data,
            bins=bins,
            range=range,
            axes_scale=axes_scale,
            weights=weights,
            color=color,
            hist_bin_factor=hist_bin_factor,
            smooth=smooth,
            smooth1d=smooth1d,
            labels=labels,
            label_kwargs=label_kwargs,
            titles=titles,
            show_titles=show_titles,
            title_quantiles=title_quantiles,
            title_fmt=title_fmt,
            title_kwargs=title_kwargs,
            truths=truths,
            truth_color=truth_color,
            scale_hist=scale_hist,
            quantiles=quantiles,
            verbose=verbose,
            fig=fig,
            max_n_ticks=max_n_ticks,
            top_ticks=top_ticks,
            use_math_text=use_math_text,
            reverse=reverse,
            labelpad=labelpad,
            hist_kwargs=hist_kwargs,
            group=group,
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
            divergences=divergences,
            divergences_kwargs=divergences_kwargs,
            labeller=labeller,
            hist2d_kwargs=hist2d_kwargs,
        )
        return

    def taylor_plot(
        self,
        ax,
        reference,
        simulations,
        Normalize=False,
        markers=[],
        colors=[],
        scale=1.5,
        markersize=2,
        pkwargs={},
        reference_name="Observations",
        simulations_name=None,
        legend=False,
        r_linewidth=1,
        r_linecolor="k",
        r_linestyle="--",
        ref_std_linewidth=1.5,
        ref_std_linecolor="k",
        ref_std_linestyle="-",
        rmse_linewidth=0.75,
        rmse_linecolor="grey",
        rmse_linestyle="--",
    ):
        TaylorDiagram(
            ax,
            reference,
            simulations,
            Normalize=Normalize,
            markers=markers,
            colors=colors,
            scale=scale,
            markersize=markersize,
            pkwargs=pkwargs,
            reference_name=reference_name,
            simulations_name=simulations_name,
            legend=legend,
            r_linewidth=r_linewidth,
            r_linecolor=r_linecolor,
            r_linestyle=r_linestyle,
            ref_std_linewidth=ref_std_linewidth,
            ref_std_linecolor=ref_std_linecolor,
            ref_std_linestyle=ref_std_linestyle,
            rmse_linewidth=rmse_linewidth,
            rmse_linecolor=rmse_linecolor,
            rmse_linestyle=rmse_linestyle,
        )


if __name__ == "__main__":
    fig = Figure()
    ax = fig.add_axes_cm("Test", 2, 1, 6, 6)
    import pandas as pd

    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 7, 6], "c": [2, 3, 4, 6, 9]}
    )

    # ax2 = fig.add_axes_cm("Demo", 3, 4, 3, 4)
    fig.taylor_plot(ax, df.iloc[:, 0].values, df.iloc[:, 1:].values)
    fig.resize(16, 8)
    fig.show()
