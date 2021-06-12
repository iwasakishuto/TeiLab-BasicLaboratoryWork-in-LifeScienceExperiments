#coding: utf-8
"""A group of plot functions useful for analysis using matplotlib_. 

If you would like to 

- make changes to the plot or draw other plots, please refer to the `official documentation <https://matplotlib.org/>`_.
- see the analysis examples, please refer to the :fa:`home` `notebook/Local/Main-Lecture-Material-matplotlib.ipynb <https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Local/Main-Lecture-Material-matplotlib.ipynb>`_

※ Compared to :doc:`teilab.plot.plotly`, you can see the difference between the two libraries. (matplotlib_, and plotly_)

.. _matplotlib: https://github.com/matplotlib/matplotlib
.. _plotly: https://github.com/plotly/plotly.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numbers import Number
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from typing import Any,Dict,List,Tuple,Optional,Union
from nptyping import NDArray
from pandas.core.series import Series

# If you want to use LateX
# from matplotlib import rc
# rc('text', usetex=True)

from ..utils.plot_utils import get_colorList, subplots_create

def density_plot(data:NDArray[(Any,Any),Number], 
                 labels:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None, 
                 bins:Union[int,List[Number],str]=100, range:Optional[Tuple[float,float]]=None, 
                 title:str="Density Distribution", ax:Optional[Axes]=None, 
                 plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Axes:
    """Plot density dirstibutions.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        bins (Union[int,List[Number],str], optional)   : The number of equal-width bins in the given range. Defaults to ``100``.
        range (Optional[Tuple[float,float]], optional) : The lower and upper range of the bins. If not provided, range is simply ``(data[i].min(), data[i].max())``. Defaults to ``None``.
        title (str, optional)                          : Figure Title. Defaults to ``"Density Distribution"``.
        ax (Optional[Axes], optional)                  : An instance of ``Axes``. Defaults to ``None``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``ax.plot``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.matplotlib.update_layout>`. Defaults to ``{}``.

    Returns:
        Axes: An instance of ``Axes`` with density distributions.

    .. plot::
        :include-source:
        :class: popup-img
        
        >>> import numpy as np
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.matplotlib import density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
        >>> nfigs = len(kwargses)
        >>> fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
        >>> for ax,kwargs in zip(axes,kwargses):
        ...     _ = density_plot(data, ax=ax, title=dict2str(kwargs), **kwargs)
        >>> fig.show()
    """
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    if data.ndim==1: data = data.reshape(1,-1)
    n_samples, n_features = data.shape
    if len(labels)!=n_samples: labels = [f"No.{i}" for i,_ in enumerate(data)]
    if len(colors)!=n_samples: colors = get_colorList(n=n_samples, cmap=cmap, style="matplotlib")
    for i,ith_data in enumerate(data):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        ax.plot(bin_edges[1:], hist, label=labels[i], color=colors[i], **plotkwargs)
    ax = update_layout(ax, title=title, **layoutkwargs, **kwargs)
    return ax

def cumulative_density_plot(data:Union[NDArray[(Any,Any),Number],Series],
                            labels:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None,
                            title:str="Cumulative Density Distribution", ylabel="Frequency", ax:Optional[Axes]=None, 
                            plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Axes:
    """Plot cumulative density dirstibutions.

    Args:
        data (Union[NDArray[(Any,Any),Number],Series]) : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        title (str, optional)                          : Subplot Title. Defaults to ``"Cumulative Density Distribution"``.
        ylabel (str, optional)                         : Subplot y-axis label. Defaults to ``"Frequency"``
        ax (Optional[Axes], optional)                  : An instance of ``Axes``. Defaults to ``None``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``ax.plot``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.matplotlib.update_layout>`. Defaults to ``{}``.

    Returns:
        Axes:An instance of ``Axes`` with cumulative density distributions.

    .. plot::
        :include-source:
        :class: popup-img
        
        >>> import numpy as np
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.matplotlib import cumulative_density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
        >>> ax = cumulative_density_plot(data, ax=ax, xlabel="value")
        >>> fig.show()
    """
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    if isinstance(data, Series): data = data.values
    if data.ndim==1: data = data.reshape(1,-1)
    data = np.sort(a=data, axis=1)
    n_samples, n_features = data.shape
    if len(labels)!=n_samples: labels = [f"No.{i}" for i,_ in enumerate(data)]
    if len(colors)!=n_samples: colors = get_colorList(n=n_samples, cmap=cmap, style="matplotlib")
    y = [(i+1)/n_features for i in range(n_features)]
    for i,ith_data in enumerate(data):
        ax.plot(ith_data, y, label=labels[i], color=colors[i], **plotkwargs)
    ax = update_layout(ax, title=title, ylabel=ylabel, **layoutkwargs, **kwargs)
    return ax

def boxplot(data:NDArray[(Any,Any),Number], 
            labels:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None, 
            vert:bool=True,
            title:str="Box Plot", ax:Optional[Axes]=None, 
            plotkwargs:Dict[str,Any]=dict(medianprops={"color": "black"}, flierprops={"marker":'o',"markersize":4,"markerfacecolor":"red","markeredgecolor":"black"}),
            layoutkwargs:Dict[str,Any]={}, **kwargs) -> Axes:
    """Plot box plots.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        vert (bool, optional)                          : Whether to draw vertical boxes or horizontal boxes. Defaults to ``True`` .
        title (str, optional)                          : Figure Title. Defaults to ``"Box Plot"``.
        ax (Optional[Axes], optional)                  : An instance of ``Axes``. Defaults to ``None``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``ax.plot`. Defaults to ``dict(medianprops={"color": "black"}, flierprops={"marker":'o',"markersize":4,"markerfacecolor":"red","markeredgecolor":"black"})``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.matplotlib.update_layout>`. Defaults to ``{}``.

    Returns:
        Axes: An instance of ``Axes`` with box plots.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.matplotlib import boxplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> kwargses = [{"vert":True},{"vert":False}]
        >>> nfigs = len(kwargses)
        >>> fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
        >>> for ax,kwargs in zip(axes,kwargses):
        ...     _ = boxplot(data, title=dict2str(kwargs), ax=ax, **kwargs)
        >>> fig.show()
    """
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    n_samples, n_features = data.shape
    if len(labels) != n_samples: labels = [f"No.{i}" for i in range(n_samples)]
    if len(colors) != n_samples: colors = get_colorList(n=n_samples, cmap=cmap, style="matplotlib")
    bplot1= ax.boxplot(
        x=data.T, vert=vert, patch_artist=True, labels=labels,
        **plotkwargs,
    )
    for i,patch in enumerate(bplot1['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_edgecolor("black")
    ax = update_layout(ax, title=title, **layoutkwargs, **kwargs)
    return ax

def XYplot(df:pd.DataFrame, x:str, y:str, logarithmic:bool=True,
           color:Optional[str]=None, size:Optional[int]=None,
           ax:Optional[Axes]=None, 
           plotkwargs:Dict[str,Any]={},  layoutkwargs:Dict[str,Any]={}, **kwargs) -> Axes:
    """XY plot.

    - x-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``X``
    - y-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``Y``

    Args:
        df (pd.DataFrame)                : DataFrame
        x (str)                          : The column name for sample ``X``.
        y (str)                          : The column name for sample ``Y``.
        logarithmic (bool)               : Whether to log the values of ``df[x]`` and ``df[y]`` 
        color (Optional[str], optional)  : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        size (Optional[str], optional)   : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        ax (Optional[Axes], optional)    : An instance of ``Axes``. Defaults to ``None``.
        plotkwargs (Dict[str,Any])       : Keyword arguments for ``ax.plot`. Defaults to ``dict(medianprops={"color": "black"}, flierprops={"marker":'o',"markersize":4,"markerfacecolor":"red","markeredgecolor":"black"})``.
        layoutkwargs (Dict[str,Any])     : Keyword arguments for :func:`update_layout <teilab.plot.matplotlib.update_layout>`. Defaults to ``{}``.

    Returns:
        Axes: An instance of ``Axes`` with XY plot.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import pandas as pd
        >>> from teilab.datasets import TeiLabDataSets
        >>> from teilab.plot.matplotlib import XYplot
        >>> from teilab.utils import subplots_create
        >>> datasets = TeiLabDataSets(verbose=False)
        >>> df_anno = datasets.read_data(no=0, usecols=datasets.ANNO_COLNAMES)
        >>> reliable_index = set(df_anno.index)
        >>> df_combined = df_anno.copy(deep=True)
        >>> for no in range(2):
        ...     df_data = datasets.read_data(no=no)
        ...     reliable_index = reliable_index & set(datasets.reliable_filter(df=df_data))
        ...     df_combined = pd.concat([
        ...         df_combined, 
        ...         df_data[[datasets.TARGET_COLNAME]].rename(columns={datasets.TARGET_COLNAME: datasets.samples.Condition[no]})
        ...     ], axis=1)
        >>> df_combined = df_combined.loc[reliable_index, :].reset_index(drop=True)
        >>> fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
        >>> ax = XYplot(df=df_combined, x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], ax=ax)
        >>> fig.show()
    """
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    df = df.copy(deep=True)
    if logarithmic:
        df[x] = df[x].apply(lambda x:np.log2(x))
        df[y] = df[y].apply(lambda y:np.log2(y))
    if color  is not None: color = df[color]
    if size   is not None: size  = df[size]
    ax.scatter(x=df[x], y=df[y], s=size, c=color,  **plotkwargs)
    ax = update_layout(
        ax=ax, 
        title=f"XY plot ({x} vs {y})", 
        xlabel=r"$\log_{2}(\mathrm{" + x + "})$", 
        ylabel=r"$\log_{2}(\mathrm{" + y + "})$", 
        **layoutkwargs, **kwargs
    )
    return ax

def MAplot(df:pd.DataFrame, x:str, y:str,
           color:Optional[str]=None, size:Optional[str]=None,
           hlines:Union[Dict[Number,Dict[str,Any]],List[Number]]=[],
           ax:Optional[Axes]=None, 
           plotkwargs:Dict[str,Any]={},  layoutkwargs:Dict[str,Any]={}, **kwargs) -> Axes:
    """MA plot.

    - x-axis (Average): :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``X``
    - y-axis (Minus)  : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``Y``

    Args:
        df (pd.DataFrame)                                       : DataFrame
        x (str)                                                 : The column name for sample ``X``.
        y (str)                                                 : The column name for sample ``Y``.
        color (Optional[str], optional)                         : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        size (Optional[str], optional)                          : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        hlines (Union[Dict[Number,Dict[str,Any]],List[Number]]) : Height (``y``) to draw a horizon. If given a dictionary, values means kwargs of ``ax.hlines``
        ax (Optional[Axes], optional)                           : An instance of ``Axes``. Defaults to ``None``.
        plotkwargs (Dict[str,Any])                              : Keyword arguments for ``ax.plot`. Defaults to ``dict(medianprops={"color": "black"}, flierprops={"marker":'o',"markersize":4,"markerfacecolor":"red","markeredgecolor":"black"})``.
        layoutkwargs (Dict[str,Any])                            : Keyword arguments for :func:`update_layout <teilab.plot.matplotlib.update_layout>`. Defaults to ``{}``.

    Returns:
        Axes: An instance of ``Axes`` with MA plot.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import pandas as pd
        >>> from teilab.datasets import TeiLabDataSets
        >>> from teilab.plot.matplotlib import MAplot
        >>> from teilab.utils import subplots_create
        >>> datasets = TeiLabDataSets(verbose=False)
        >>> df_anno = datasets.read_data(no=0, usecols=datasets.ANNO_COLNAMES)
        >>> reliable_index = set(df_anno.index)
        >>> df_combined = df_anno.copy(deep=True)
        >>> for no in range(2):
        ...     df_data = datasets.read_data(no=no)
        ...     reliable_index = reliable_index & set(datasets.reliable_filter(df=df_data))
        ...     df_combined = pd.concat([
        ...         df_combined, 
        ...         df_data[[datasets.TARGET_COLNAME]].rename(columns={datasets.TARGET_COLNAME: datasets.samples.Condition[no]})
        ...     ], axis=1)
        >>> df_combined = df_combined.loc[reliable_index, :].reset_index(drop=True)
        >>> fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
        >>> ax = MAplot(
        ...     df=df_combined, 
        ...     x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], ax=ax,
        ...     hlines={
        ...         -1 : dict(colors='r', linewidths=1),
        ...          0 : dict(colors='r', linewidths=2),
        ...          1 : dict(colors='r', linewidths=1),
        ...     }
        >>> )
        >>> fig.show()    
    """
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    df = df.copy(deep=True)
    X = np.log10(df[x]*df[y])
    Y = np.log2(df[y]/df[x])
    if color  is not None: color = df[color]
    if size   is not None: size  = df[size]
    ax.scatter(x=X, y=Y, s=size, c=color,  **plotkwargs)
    if isinstance(hlines, list): hlines = dict(zip(hlines, [{} for _ in range(len(hlines))]))
    for key,hlineskw in hlines.items():
        ax.hlines(y=key, xmin=min(X), xmax=max(X), **hlineskw)
    ax = update_layout(
        ax=ax, 
        title=f"MA plot ({x} vs {y})", 
        xlabel=r"$\log_{10}(\mathrm{" + x + r"}\times\mathrm{" + y + "})$", 
        ylabel=r"$\log_{2}(\mathrm{" + y + r"} / \mathrm{" + x + "})$", 
        **layoutkwargs, **kwargs
    )
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

    .. plot::
        :include-source:
        :class: popup-img

        >>> from teilab.utils import subplots_create
        >>> from teilab.plot.matplotlib import update_layout
        >>> fig, axes = subplots_create(ncols=2, style="matplotlib", figsize=(8,4))
        >>> for ax in axes: ax.scatter(1,1,label="center")
        >>> _ = update_layout(ax=axes[1], xlim=(0,2), ylim=(0,2), legend=True)
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