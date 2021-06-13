#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from typing import Optional,Any,Tuple
from numbers import Number
from matplotlib.axes import Axes
from nptyping import NDArray

from .utils.plot_utils import subplots_create
from .plot.matplotlib import update_layout

# def ttest_rel()

# Right-tail	Ha:σ21>σ22	F≥Fα
# Left-tail	Ha:σ21<σ22	F≤F1−α
# Two-tail

def f_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, tail:str="two", plot:bool=False, ax:Optional[Axes]=None) -> bool:
    """F-Tests for Equality of Two Variances.

    If the two populations are normally distributed and if :math:`H_0:\sigma^2_1=\sigma^2_2` is true then under independent sampling :math:`F` approximately follows an F-distribution (:math:`f(x, df_1, df_2)`) with degrees of freedom :math:`df_1=n_1−1` and :math:`df_2=n_2−1`.

    The probability density function for `f` is:
    
    .. math::

        f(x, df_1, df_2) = \\frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}

    for :math:`x > 0`.

    +--------------------------+-------------------------------------------+---------------------------------------------------------------------------------+
    | Terminology (``tail``)   | Alternative Hypothesis                    | Rejection Region                                                                |
    +==========================+===========================================+=================================================================================+
    | ``"right"``-tailed       | :math:`H_a:\sigma^2_1   >  \sigma^2_2`    | :math:`\mathbf{F}\geq F_{\\alpha}`                                               |
    +--------------------------+-------------------------------------------+---------------------------------------------------------------------------------+
    | ``"left"``-tailed        | :math:`H_a:\sigma^2_1   <  \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\\alpha}`                                             |
    +--------------------------+-------------------------------------------+---------------------------------------------------------------------------------+
    | ``"two"``-tailed         | :math:`H_a:\sigma^2_1 \\neq \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\\alpha∕2}` or :math:`\mathbf{F}\geq F_{\\alpha∕2}`    |
    +--------------------------+-------------------------------------------+---------------------------------------------------------------------------------+

    Args:
        a (NDArray[Any, Number])      : [description]
        b (NDArray[Any, Number])      : [description]
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``True``.
        alpha (float, optional)       : [description]. Defaults to ``0.05``.
        tail (str, optional)          : [description]. Defaults to ``"two"``.
        ax (Optional[Axes], optional) : [description]. Defaults to ``None``.

    Returns:
        bool: Does the null hypothesis hold?

    .. plot::
        :include-source:
        :class: popup-img
        
        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import f_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
        >>> B = np.array([4.8, 2.1, 5.1, 2.0, 4.0, 1.0, 3.4, 2.7, 5.1, 1.4, 1.6])
        >>> for ax,tail in zip(axes,["left","two","right"]):
        ...     f_test(A, B, tail=tail, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::
        https://en.wikipedia.org/wiki/F-test    
    """
    a_unbiased_var = np.var(a, ddof=1) #: ``a``'s unbiased variable. 
    b_unbiased_var = np.var(b, ddof=1) #: ``b``'s unbiased variable. 
    a_degrees_of_freedom = len(a)-1 #: ``a``'s Degrees of Freedom.
    b_degrees_of_freedom = len(b)-1 #: ``b``'s Degrees of Freedom.
    f = a_unbiased_var / b_unbiased_var    
    if tail == "left":
        x_l = stats.f.ppf(alpha,   dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
        x_r = stats.f.ppf(1,       dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
    elif tail == "right":
        x_l = stats.f.ppf(0,       dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
        x_r = stats.f.ppf(1-alpha, dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
    else: # Two-side
        x_l = stats.f.ppf(alpha/2,   dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
        x_r = stats.f.ppf(1-alpha/2, dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)

    ret = x_l < f < x_r
    if plot:
        ax = ax or subplots_create(style="matplotlib")[1]
        x_max = 3
        while True:
            if stats.f.pdf(x_max, dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)<1e-3: break
            x_max+=1
        x = np.linspace(0.001,max(x_max,f)+1,1000)
        y = stats.f.pdf(x, dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
        # F-distribution
        ax.plot(x, y, color="blue", label=fr"f-distribution $f(x,dfn,dfd)$ (dfn={a_degrees_of_freedom}, dfd={b_degrees_of_freedom})")
        # Indicates the position of calculated "F"
        f_y = stats.f.pdf(f, dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom)
        ax.scatter(x=f, y=f_y, color="black", label=fr"$(x=F={f:.2f},y={f_y:.2f})$")
        # Visualize the Right and Left rejection areas
        ax.fill_between(x=x, y1=0, y2=y, where=(x<x_l), color="red", label=f"Rejection Region (alpha={alpha})")
        ax.fill_between(x=x, y1=0, y2=y, where=(x>x_r), color="red")
        ax = update_layout(ax=ax, title=f"F-test ({tail.capitalize()}-tailed)\nNull-hypothesis {'holds' if ret else 'rejects'}", xlabel="x", ylabel="y", legend=True)
    return x_l < f < x_r
