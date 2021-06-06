#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from typing import Any,Dict,List,Tuple,Optional,Union
from nptyping import NDArray

def _ax_create(ax:Optional[Axes]=None, figsize:Tuple[Number,Number]=(6,4)) -> Axes:
    """Create an ``Axes`` instance.

    Args:
        ax (Optional[Axes], optional)            : An ``Axes`` instance. Defaults to ``None``.
        figsize (Tuple[Number,Number], optional) : Figure size if you create a new figure. deman. Defaults to ``(6,4)``.

    Returns:
        Axes: An instance of ``Axes``
    """
    if ax is None:
        _,ax = plt.subplots(figsize=figsize)
    return ax

def densityplot(data:NDArray[(Any,Any),Number], 
                names:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None, 
                bins:Union[int,List[Number],str]=100, range:Optional[Tuple[float,float]]=None, 
                title:str="Density Distribution", ax:Optional[Axes]=None, **kwargs) -> Axes:
    """Plot density dirstibutions.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        names (List[str], optional)                    : Names for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        bins (Union[int,List[Number],str], optional)   : The number of equal-width bins in the given range. Defaults to ``100``.
        range (Optional[Tuple[float,float]], optional) : The lower and upper range of the bins. If not provided, range is simply ``(data[i].min(), data[i].max())``. Defaults to ``None``.
        title (str, optional)                          : Figure Title. Defaults to ``"Density Distribution"``.
        ax (Optional[Axes], optional)                  : An instance of ``Axes``. Defaults to ``None``.

    Returns:
        Axes: An instance of ``Axes`` with density distributions.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from teilab.utils import dict2str
        >>> from teilab.plot.matplotlib import densityplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> kwarges = [{"bins":100},{"bins":10},{"bins":"auto"}]
        >>> nfigs = len(kwarges)
        >>> fig, axes = plt.subplots(ncols=nfigs, figsize=(int(6*nfigs),4))
        >>> for ax,kwargs in zip(axes,kwarges):
        ...     _ = densityplot(data, ax=ax, title=dict2str(kwargs), **kwargs)
        >>> fig.show()

    +----------------------------------------------------+
    |                      Results                       |
    +====================================================+
    | .. image:: _images/plot.matplotlib.densityplot.jpg |
    |    :class: popup-img                               |
    +----------------------------------------------------+
    """
    ax = _ax_create(ax=ax)
    names = names or [f"No.{i}" for i,_ in enumerate(data)]
    for ith_data,name in zip(data,names):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        ax.plot(bin_edges[1:], hist, label=name)
    ax = update_layout(ax, title=title, **kwargs)
    return ax

def boxplot(data:NDArray[(Any,Any),Number], 
            names:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None, 
            vert:bool=True,
            title:str="Box Plot", ax:Optional[Axes]=None, **kwargs) -> Axes:
    """Plot box plots.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        names (List[str], optional)                    : Names for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        vert (bool, optional)                          : Whether to draw vertical boxes or horizontal boxes. Defaults to ``True`` .
        title (str, optional)                          : Figure Title. Defaults to ``"Box Plot"``.
        ax (Optional[Axes], optional)                  : An instance of ``Axes``. Defaults to ``None``.

    Returns:
        Axes: An instance of ``Axes`` with box plots.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from teilab.utils import dict2str
        >>> from teilab.plot.matplotlib import boxplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> kwarges = [{"vert":True},{"vert":False}]
        >>> nfigs = len(kwarges)
        >>> fig, axes = plt.subplots(ncols=nfigs, figsize=(int(6*nfigs),4))
        >>> for ax,kwargs in zip(axes,kwarges):
        ...     _ = boxplot(data, title=dict2str(kwargs), ax=ax, **kwargs)
        >>> fig.show()

    +------------------------------------------------+
    |                    Results                     |
    +================================================+
    | .. image:: _images/plot.matplotlib.boxplot.jpg |
    |    :class: popup-img                           |
    +------------------------------------------------+
    """
    ax = _ax_create(ax=ax)
    n_samples, n_features = data.shape
    if len(names) != n_samples: names = [f"No.{i}" for i in range(n_samples)]
    if len(colors)!= n_samples: 
        cmap = plt.get_cmap(name=cmap)
        colors = [cmap(((i+1)/(n_samples))) for i in range(n_samples)]
    bplot1= ax.boxplot(
        x=data.T, vert=vert, patch_artist=True, labels=names,
        medianprops={"color": "black"},
        flierprops={"marker":'o',"markersize":4,"markerfacecolor":"red","markeredgecolor":"black"},
    )
    for i,patch in enumerate(bplot1['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor("black")
    ax = update_layout(ax, title=title, **kwargs)
    return ax

def XYplot(X, Y, color=None, marker_size=3, legend=True, ax=None,
           xlabel="$\log_{2}X$", ylabel="$\log_{2}(Y)$", title="XY plot", **kwargs):
    ax = _ax_create(ax=ax)
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
    ax = _ax_create(ax=ax)
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
                  xlim:List[float]=[], ylim:List[float]=[], legend:bool=True) -> Axes:
    """Update the layout of ``matplotlib.axes.Axes`` object. See `Documentation <https://matplotlib.org/stable/api/axes_api.html>`_ for details.

    Args:
        ax (Axes)                         : An instance of ``Axes``.
        title (Optional[str], optional)   : Figure title. Defaults to ``None``.
        xlabel (Optional[str], optional)  : X axis label. Defaults to ``None``.
        ylabel (Optional[str], optional)  : Y axis label. Defaults to ``None``.
        xlim (List[float])                : X axis range. Defaults to ``[]``.
        ylim (List[float])                : Y axis range. Defaults to ``[]``.
        legend (bool, optional)           : Whether to show legend. Defaults to ``True``.

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
    if legend and len(ax.get_legend_handles_labels()[1])>0:
        ax.legend()
    return ax