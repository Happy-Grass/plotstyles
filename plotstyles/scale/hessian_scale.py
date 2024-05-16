from matplotlib.scale import ScaleBase, register_scale
from matplotlib import _api, _docstring
from matplotlib.transforms import Transform
from scipy.stats import norm
from matplotlib.ticker import (
    Formatter, Locator, NullFormatter, ScalarFormatter, NullLocator,  AutoLocator, AutoMinorLocator)
import numpy as np
import matplotlib as mpl


class HessianTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base_p):
        super().__init__()
        if base_p <= 0 :
            raise ValueError('The hessian base_p cannot be <= 0')
        self.base_p = base_p

    def __str__(self):
        return "{}(base_p={}".format(
            type(self).__name__, self.base_p)

    def transform_non_affine(self, a):
        return norm.ppf(a) - norm.ppf(self.base_p)


    def inverted(self):
        return InvertedHessianTransform(self.base_p)
    
class InvertedHessianTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base_p):
        super().__init__()
        self.base_p = base_p

    def __str__(self):
        return "{}(base_p={})".format(type(self).__name__, self.base_p)

    def transform_non_affine(self, a):
        return norm.cdf(a + norm.ppf(self.base_p))

    def inverted(self):
        return HessianTransform(self.base_p)
    
class HessianScale(ScaleBase):
    """
    Hessian Scale was used to arrange the probability values 
    on the coordinate axis according to the normal distribution
    """
    name = 'hessian'

    def __init__(self, axis, base_p=0.0001, subs=None):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        base_p : float, default: 0.01
            The base of the probability.
        subs : sequence of int, default: None
            Where to place the subticks between each major tick.  For example,
            in a log10 scale, ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place 8
            logarithmically spaced minor ticks between each major tick.
        """
        self._transform = HessianTransform(base_p)
        self.subs = subs
    base_p = property(lambda self: self._transform.base_p)

    def get_transform(self):
        """Return the `.HessianTransform` associated with this scale."""
        return self._transform
    
    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # update the minor locator for x and y axis based on rcParams
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())

register_scale(HessianScale)  # 注册自定义scale   