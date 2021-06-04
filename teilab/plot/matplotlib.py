#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any,Dict,List,Tuple,Optional

def _ax_create(ax=None, figsize=(6,4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax

def plotDensities(data, names=None, bins=100, range=None, ax=None, **kwargs):
    ax = _ax_create(ax=ax)
    names = names or [f"data_{i+1}" for i in range(len(data))]
    for ith_data,name in zip(data,names):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        ax.plot(bin_edges[1:], hist, label=name)
    ax = update_layout(ax, **kwargs)
    return ax

def XYplot(X, Y, color=None, marker_size=3, legend=True, ax=None,
           xlabel="$\log_{2}X$", ylabel="$\log_{2}(Y)$", title="XY plot", **kwargs):
    ax = _ax_create(ax)
    color = np.zeros_like(X) if color is None else color
    x = np.log2(X)
    y = np.log2(Y)
    for c in np.unique(color):
        idx = color==c
        ax.scatter(x=x[idx], y=y[idx], label=str(c), s=marker_size)
    ax = update_layout(ax, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend, **kwargs)
    return ax

def MAplot(X, Y, color=None, marker_size=3, legend=True, ax=None,
           xlabel="$\log_{10}XY$", ylabel="$\log_{2}(Y/X)$", title="MA plot", **kwargs):
    ax = _ax_create(ax)
    color = np.zeros_like(X) if color is None else color
    x = np.log10(X*Y)
    y = np.log2(Y/X)
    for c in np.unique(color):
        idx = color==c
        ax.scatter(x=x[idx], y=y[idx], label=str(c), s=marker_size)
    ax = update_layout(ax, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend, **kwargs)
    return ax

def update_layout(ax:Axes, 
                  title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None, 
                  xlim:Optional[List[float]]=None, ylim:Optional[List[float]]=None, 
                  legend:bool=True) -> Axes:
    """Update the layout of ``matplotlib.axes.Axes`` object. See `Documentation <https://matplotlib.org/stable/api/axes_api.html>`_ for details.

    Args:
        ax (Axes)                              : A figure element.
        title (Optional[str], optional)        : Figure title. Defaults to ``None``.
        xlabel (Optional[str], optional)       : X axis label. Defaults to ``None``.
        ylabel (Optional[str], optional)       : Y axis label. Defaults to ``None``.
        xlim (Optional[List[float]], optional) : X axis range. Defaults to ``None``.
        ylim (Optional[List[float]], optional) : Y axis range. Defaults to ``None``.
        legend (bool, optional)                : Whether to show legend. Defaults to ``True``.

    Returns:
        Axes: A figure element with layout updated.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from teilab.plot.matplotlib import update_layout
        >>> fig, ax = plt.subplots()
        >>> ax.scatter(1,1,label="center")
        >>> fig.show()
        >>> ax = update_layout(ax=ax, xlim=(0,2), ylim=(0,2), legend=True)
        >>> fig.show()
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if legend:
        ax.legend()
    return ax