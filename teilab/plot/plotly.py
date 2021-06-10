#coding: utf-8
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numbers import Number
from plotly.graph_objs import Figure
from matplotlib.colors import Colormap
from typing import Any,Dict,List,Tuple,Optional,Union
from nptyping import NDArray
from pandas.core.series import Series

from ..utils.plot_utils import get_colorList

def density_plot(data:NDArray[(Any,Any),Number],
                 labels:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None,
                 bins:int=100, range:Optional[Tuple[float,float]]=None, 
                 title:str="Density Distribution", fig:Optional[Figure]=None, row:int=1, col:int=1, 
                 plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Figure:
    """Plot density dirstibutions.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        bins (int, optional)                           : The number of equal-width bins in the given range. Defaults to ``100``.
        range (Optional[Tuple[float,float]], optional) : The lower and upper range of the bins. If not provided, range is simply ``(data[i].min(), data[i].max())``. Defaults to ``None``.
        title (str, optional)                          : Figure Title. Defaults to ``"Density Distribution"``.
        fig (Optional[Figure], optional)               : An instance of Figure.
        row (int, optional)                            : Row of subplots. Defaults to ``1``.
        col (int, optional)                            : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``go.Scatter``
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>` .

    Returns:
        Figure: An instance of ``Figure`` with density distributions.
    
    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import matplotlib.pyplot as plt
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.plotly import density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
        >>> title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
        >>> nfigs = len(kwargses)
        >>> fig = subplots_create(ncols=nfigs, style="plotly")
        >>> for i,(kwargs) in enumerate(kwargses, start=1):
        ...     _ = density_plot(data, fig=fig, title=title, col=i, legend=False, width=1000, height=400, **kwargs)
        >>> fig.show()
    """
    fig = fig or make_subplots(rows=1, cols=1)
    if data.ndim==1: data = data.reshape(1,-1)
    n_samples, n_features = data.shape
    if len(labels)!=n_samples: labels = [f"No.{i}" for i,_ in enumerate(data)]
    if len(colors)!=n_samples: colors = get_colorList(n=n_samples, cmap=cmap, style="plotly")
    for i,ith_data in enumerate(data):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        fig.add_trace(trace=go.Scatter(x=bin_edges[1:], y=hist, name=labels[i], mode="lines", fillcolor=colors[i], marker={"color":colors[i]}, **plotkwargs), row=row, col=col)
    fig = update_layout(fig, row=row, col=col, title=title, **layoutkwargs, **kwargs)
    return fig

def cumulative_density_plot(data:Union[NDArray[(Any,Any),Number],Series],
                            labels:List[str]=[], colors:List[Any]=[], cmap:Optional[Union[str,Colormap]]=None,
                            title:str="Cumulative Density Distribution", ylabel="Frequency",
                            fig:Optional[Figure]=None, row:int=1, col:int=1, 
                            plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Figure:
    """Plot cumulative density dirstibutions.

    Args:
        data (Union[NDArray[(Any,Any),Number],Series]) : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        title (str, optional)                          : Figure Title. Defaults to ``"Cumulative Density Distribution"``.
        ylabel (str, optional)                         : Figure y-axis label. Defaults to ``"Frequency"``
        fig (Optional[Figure], optional)               : An instance of Figure.
        row (int, optional)                            : Row of subplots. Defaults to ``1``.
        col (int, optional)                            : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``go.Scatter``
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>` .

    Returns:
        Figure: An instance of ``Figure`` with density distributions.
    
    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import matplotlib.pyplot as plt
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.plotly import cumulative_density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
        >>> fig = cumulative_density_plot(data, fig=None, xlabel="value", width=800, height=400)
        >>> fig.show()
    """
    fig = fig or make_subplots(rows=1, cols=1)
    if isinstance(data, Series): data = data.values
    if data.ndim==1: data = data.reshape(1,-1)
    data = np.sort(a=data, axis=1)
    n_samples, n_features = data.shape
    if len(labels)!=n_samples: labels = [f"No.{i}" for i,_ in enumerate(data)]
    if len(colors)!=n_samples: colors = get_colorList(n=n_samples, cmap=cmap, style="plotly")
    y = [(i+1)/n_features for i in range(n_features)]
    for i,ith_data in enumerate(data):
        fig.add_trace(trace=go.Scatter(x=ith_data, y=y, name=labels[i], mode="lines", fillcolor=colors[i], marker={"color":colors[i]}, **plotkwargs), row=row, col=col)
    fig = update_layout(fig, row=row, col=col, title=title, ylabel=ylabel, **layoutkwargs, **kwargs)
    return fig

def XYplot(df:pd.DataFrame, x:str, y:str, logarithmic:bool=True,
           color:Optional[str]=None, symbol:Optional[str]=None, size:Optional[str]=None, 
           hover_data:Optional[List[str]]=None, hover_name:Optional[str]=None, 
           fig:Optional[Figure]=None, row:int=1, col:int=1,
           plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Figure:
    """XY plot.

    - x-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``X``
    - y-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``Y``

    Args:
        df (pd.DataFrame)                          : DataFrame
        x (str)                                    : The column name for sample ``X``.
        y (str)                                    : The column name for sample ``Y``.
        color (Optional[str], optional)            : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        symbol (Optional[str], optional)           : The column name in ``df`` to assign symbols to marks. Defaults to ``None``.
        size (Optional[str], optional)             : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        hover_data (Optional[List[str]], optional) : Values in this column appear in bold in the hover tooltip. Defaults to ``None``.
        hover_name (Optional[str], optional)       : Values in this column appear in the hover tooltip. Defaults to ``None``.
        fig (Optional[Figure], optional)           : An instance of Figure.
        row (int, optional)                        : Row of subplots. Defaults to ``1``.
        col (int, optional)                        : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                 : Keyword arguments for ``go.Scatter``
        layoutkwargs (Dict[str,Any])               : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>` .

    Returns:
        Figure: An instance of ``Figure`` with XY plot.

    .. plotly::
        :include-source:
        :iframe-height: 600px

        >>> from teilab.datasets import TeiLabDataSets
        >>> from teilab.plot.plotly import XYplot
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
        >>> fig = XYplot(df=df_combined, x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], hover_name="SystematicName", height=600, width=600)
        >>> fig.show()
    """
    fig = fig or make_subplots(rows=1, cols=1)
    df = df.copy(deep=True)
    if logarithmic:
        df[x] = df[x].apply(lambda x:np.log2(x))
        df[y] = df[y].apply(lambda y:np.log2(y))
    fig_ = px.scatter(data_frame=df, x=x, y=y, color=color, symbol=symbol, size=size, hover_name=hover_name, hover_data=hover_data, **plotkwargs)
    fig_.for_each_trace(fn=lambda trace:fig.add_trace(trace=trace, row=row, col=col))
    fig = update_layout(
        fig=fig, 
        title=f"XY plot ({x} vs {y})", 
        xlabel="$\\log_{2}(\\text{" + x + "})$", 
        ylabel="$\\log_{2}(\\text{" + y + "})$", 
        **layoutkwargs, **kwargs
    )
    return fig

def MAplot(df:pd.DataFrame, x:str, y:str,
           color:Optional[str]=None, symbol:Optional[str]=None, size:Optional[str]=None,
           hover_data:Optional[List[str]]=None, hover_name:Optional[str]=None,
           fig:Optional[Figure]=None, row:int=1, col:int=1,
           plotkwargs:Dict[str,Any]={}, layoutkwargs:Dict[str,Any]={}, **kwargs) -> Figure:
    """MA plot.

    - x-axis : :math:`\\log_{10}{\\left(\\text{gProcessedSignal}_X\\times\\text{gProcessedSignal}_Y\\right)}`
    - y-axis : :math:`\\log_{2}{\\left(\\text{gProcessedSignal}_Y / \\text{gProcessedSignal}_X\\right)}`

    Args:
        df (pd.DataFrame)                          : DataFrame
        x (str)                                    : The column name for sample ``X``.
        y (str)                                    : The column name for sample ``Y``.
        color (Optional[str], optional)            : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        symbol (Optional[str], optional)           : The column name in ``df`` to assign symbols to marks. Defaults to ``None``.
        size (Optional[str], optional)             : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        hover_data (Optional[List[str]], optional) : Values in this column appear in bold in the hover tooltip. Defaults to ``None``.
        hover_name (Optional[str], optional)       : Values in this column appear in the hover tooltip. Defaults to ``None``.
        fig (Optional[Figure], optional)           : An instance of Figure.
        row (int, optional)                        : Row of subplots. Defaults to ``1``.
        col (int, optional)                        : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                 : Keyword arguments for ``go.Scatter``
        layoutkwargs (Dict[str,Any])               : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>` .

    Returns:
        Figure: An instance of ``Figure`` with MA plot.

    .. plotly::
        :include-source:
        :iframe-height: 600px

        >>> from teilab.datasets import TeiLabDataSets
        >>> from teilab.plot.plotly import MAplot
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
        >>> fig = MAplot(df=df_combined, x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], hover_name="SystematicName", height=600, width=600)
        >>> fig.show()
    """
    fig = fig or make_subplots(rows=1, cols=1)
    df = df.copy(deep=True)
    x_axis_colname = "x-axis"
    y_axis_colname = "y-axis"
    df[x_axis_colname] = np.log10(df[x]*df[y])
    df[y_axis_colname] = np.log2(df[y]/df[x])
    fig_ = px.scatter(data_frame=df, x=x_axis_colname, y=y_axis_colname, color=color, symbol=symbol, size=size, hover_name=hover_name, hover_data=hover_data, **plotkwargs)
    fig_.for_each_trace(fn=lambda trace:fig.add_trace(trace=trace, row=row, col=col))
    fig = update_layout(
        fig=fig, 
        title=f"MA plot ({x} vs {y})", 
        xlabel="$\\log_{10}(\\text{" + x + "}\\times\\text{" + y + "})$", 
        ylabel="$\\log_{2}(\\left(\\text{" + y + "} / \\text{" + x + "}\\right))$", 
        **layoutkwargs, **kwargs
    )
    return fig

def update_layout(fig:Figure, row:int=1, col:int=1, 
                  title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None, 
                  xlim:Optional[Tuple[Any,Any]]=None, ylim:Optional[Tuple[Any,Any]]=None, 
                  xrangeslider:Dict[str,Any]={"visible": False},
                  font:Dict[str,Any]={"family":"verdana, arial, sans-serif", "size": 14}, legend:bool=True,
                  xaxis_type:str="linear", yaxis_type:str="linear", 
                  width:int=600, height:int=600, template:str="presentation") -> Figure:
    """Update the layout of ``plotly.graph_objs._figure.Figure`` object. See `Documentation <https://plotly.com/python/reference/layout/>`_ for details.

    Args:
        fig (Figure)                              : A Figure object.
        row (int, optional)                       : Row of subplots. Defaults to ``1``.
        col (int, optional)                       : Column of subplots. Defaults to ``1``.
        title (Optional[str], optional)           : Figure title. Defaults to ``None``.
        xlabel (Optional[str], optional)          : X axis label. Defaults to ``None``.
        ylabel (Optional[str], optional)          : Y axis label. Defaults to ``None``.
        xlim (Optional[Tuple[Any,Any]], optional) : X axis range. Defaults to ``None``.
        ylim (Optional[Tuple[Any,Any]], optional) : Y axis range. Defaults to ``None``.
        xrangeslider (Dict[str,Any], optional)    : Keyword arguments for Range Slider. Defaults to ``{"visible": False}``.
        font (Dict[str,Any], optional)            : Keyword arguments for Fonts. Defaults to ``{"family": "Meiryo", "size": 20}``.
        legend (bool, optional)                   : Whether to show legend. Defaults to ``True``.
        xaxis_type (str, optional)                : X axis type. Defaults to ``"linear"``.
        yaxis_type (str, optional)                : Y axis type. Defaults to ``"linear"``.
        width (int, optional)                     : Figure width. Defaults to ``600``.
        height (int, optional)                    : Figure height. Defaults to ``600``.
        template (str, optional)                  : Plotly themes. You can check the all template by ``python -c "import plotly.io as pio; print(pio.templates)``. Defaults to ``"plotly_white"``.

    Returns:
        Figure: Figure object with layout updated.

    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import  plotly.graph_objects as go
        >>> from plotly.subplots import make_subplots
        >>> from teilab.plot.plotly import update_layout
        >>> fig = make_subplots(rows=1, cols=2)
        >>> for c in range(1,3): fig.add_trace(go.Scatter(x=[1,2,3],y=[4,5,6]),row=1,col=c)
        >>> fig = update_layout(fig=fig, title="Sample", ylim=(4.5,5.5), col=2, height=400)
        >>> fig.show()
    """
    fig.update_xaxes(title=xlabel, range=xlim, rangeslider=xrangeslider, row=row, col=col)
    fig.update_yaxes(title=ylabel, range=ylim, row=row, col=col)
    fig.update_layout(
        title=title, font=font, width=width, height=height, template=template,
        showlegend=legend, xaxis_type=xaxis_type, yaxis_type=yaxis_type,
    )
    return fig
