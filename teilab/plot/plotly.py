# coding: utf-8
"""A group of plot functions useful for analysis using plotly_.

If you would like to

- make changes to the plot or draw other plots, please refer to the `official documentation <https://plotly.com/>`_.
- see the analysis examples, please refer to the :fa:`home` `notebook/Local/Main-Lecture-Material-plotly.ipynb <https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Local/Main-Lecture-Material-plotly.ipynb>`_

※ Compared to :doc:`teilab.plot.matplotlib`, you can see the difference between the two libraries. (matplotlib_, and plotly_)

.. _matplotlib: https://github.com/matplotlib/matplotlib
.. _plotly: https://github.com/plotly/plotly.py
"""
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nptyping import NDArray
from pandas.core.series import Series

import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import Colormap
from plotly.graph_objs import Figure

from ..utils.plot_utils import get_colorList, subplots_create, trace_transition


def density_plot(
    data: NDArray[(Any, Any), Number],
    labels: List[str] = [],
    colors: List[Any] = [],
    cmap: Optional[Union[str, Colormap]] = None,
    bins: int = 100,
    range: Optional[Tuple[float, float]] = None,
    title: str = "Density Distribution",
    fig: Optional[Figure] = None,
    row: int = 1,
    col: int = 1,
    plotkwargs: Dict[str, Any] = {},
    layoutkwargs: Dict[str, Any] = {},
    **kwargs,
) -> Figure:
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
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``go.Scatter``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>`. Defaults to ``{}``.

    Returns:
        Figure: An instance of ``Figure`` with density distributions.

    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import numpy as np
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.plotly import density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples, n_features))
        >>> kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
        >>> title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
        >>> nfigs = len(kwargses)
        >>> fig = subplots_create(ncols=nfigs, style="plotly")
        >>> for i,(kwargs) in enumerate(kwargses, start=1):
        ...     _ = density_plot(data, fig=fig, title=title, col=i, legend=False, width=1000, height=400, **kwargs)
        >>> fig.show()
    """
    fig = fig or subplots_create(nrows=1, ncols=1, style="plotly")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_samples, n_features = data.shape
    if len(labels) != n_samples:
        labels = [f"No.{i}" for i, _ in enumerate(data)]
    if len(colors) != n_samples:
        colors = get_colorList(n=n_samples, cmap=cmap, style="plotly")
    for i, ith_data in enumerate(data):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        fig.add_trace(
            trace=go.Scatter(
                x=bin_edges[1:],
                y=hist,
                name=labels[i],
                mode="lines",
                fillcolor=colors[i],
                marker={"color": colors[i]},
                legendgroup=f"{col}-{row}",
                **plotkwargs,
            ),
            row=row,
            col=col,
        )
    fig = update_layout(fig, row=row, col=col, title=title, **layoutkwargs, **kwargs)
    return fig


def boxplot(
    data: NDArray[(Any, Any), Number],
    labels: List[str] = [],
    colors: List[Any] = [],
    cmap: Optional[Union[str, Colormap]] = None,
    vert: bool = True,
    title: str = "Box Plot",
    fig: Optional[Figure] = None,
    row: int = 1,
    col: int = 1,
    plotkwargs: Dict[str, Any] = {},
    layoutkwargs: Dict[str, Any] = {},
    **kwargs,
) -> Figure:
    """Plot box plots.

    Args:
        data (NDArray[(Any,Any),Number])               : Input data. Shape = ( ``n_samples``, ``n_features`` )
        labels (List[str], optional)                   : Labels for each sample. Defaults to ``[]``.
        colors (List[Any], optional)                   : Colors for each sample. Defaults to ``[]``.
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        vert (bool, optional)                          : Whether to draw vertical boxes or horizontal boxes. Defaults to ``True`` .
        title (str, optional)                          : Figure Title. Defaults to ``"Box Plot"``.
        fig (Optional[Figure], optional)               : An instance of Figure.
        row (int, optional)                            : Row of subplots. Defaults to ``1``.
        col (int, optional)                            : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``go.Box``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>`. Defaults to ``{}``.

    Returns:
        Figure: An instance of ``Figure`` with box plot

    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import numpy as np
        >>> from teilab.utils import dict2str, subplots_create
        >>> from teilab.plot.plotly import boxplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples, n_features))
        >>> kwargses = [{"vert":True},{"vert":False}]
        >>> title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
        >>> nfigs = len(kwargses)
        >>> fig = subplots_create(ncols=nfigs, style="plotly")
        >>> for col,kwargs in enumerate(kwargses, start=1):
        ...     _ = boxplot(data, title=title, fig=fig, col=col, width=1000, height=400, **kwargs)
        >>> fig.show()
    """
    fig = fig or subplots_create(nrows=1, ncols=1, style="plotly")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_samples, n_features = data.shape
    if len(labels) != n_samples:
        labels = [f"No.{i}" for i in range(n_samples)]
    if len(colors) != n_samples:
        colors = get_colorList(n=n_samples, cmap=cmap, style="plotly")
    key = "y" if vert else "x"
    for i, ith_data in enumerate(data):
        fig.add_trace(
            trace=go.Box(
                **{key: ith_data}, name=labels[i], marker_color=colors[i], legendgroup=f"{col}-{row}", **plotkwargs
            ),
            row=row,
            col=col,
        )
    fig = update_layout(fig, row=row, col=col, title=title, **layoutkwargs, **kwargs)
    return fig


def cumulative_density_plot(
    data: Union[NDArray[(Any, Any), Number], Series],
    labels: List[str] = [],
    colors: List[Any] = [],
    cmap: Optional[Union[str, Colormap]] = None,
    title: str = "Cumulative Density Distribution",
    ylabel="Frequency",
    fig: Optional[Figure] = None,
    row: int = 1,
    col: int = 1,
    plotkwargs: Dict[str, Any] = {},
    layoutkwargs: Dict[str, Any] = {},
    **kwargs,
) -> Figure:
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
        plotkwargs (Dict[str,Any])                     : Keyword arguments for ``go.Scatter``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                   : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>`. Defaults to ``{}``.

    Returns:
        Figure: An instance of ``Figure`` with cumulative density distributions.

    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import numpy as np
        >>> from teilab.plot.plotly import cumulative_density_plot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples, n_features))
        >>> fig = cumulative_density_plot(data, fig=None, xlabel="value", width=800, height=400)
        >>> fig.show()
    """
    fig = fig or subplots_create(nrows=1, ncols=1, style="plotly")
    if isinstance(data, Series):
        data = data.values
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = np.sort(a=data, axis=1)
    n_samples, n_features = data.shape
    if len(labels) != n_samples:
        labels = [f"No.{i}" for i, _ in enumerate(data)]
    if len(colors) != n_samples:
        colors = get_colorList(n=n_samples, cmap=cmap, style="plotly")
    y = [(i + 1) / n_features for i in range(n_features)]
    for i, ith_data in enumerate(data):
        fig.add_trace(
            trace=go.Scatter(
                x=ith_data,
                y=y,
                name=labels[i],
                mode="lines",
                fillcolor=colors[i],
                marker={"color": colors[i]},
                legendgroup=f"{col}-{row}",
                **plotkwargs,
            ),
            row=row,
            col=col,
        )
    fig = update_layout(fig, row=row, col=col, title=title, ylabel=ylabel, **layoutkwargs, **kwargs)
    return fig


def XYplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    logarithmic: bool = True,
    color: Optional[str] = None,
    symbol: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    hover_name: Optional[str] = None,
    fig: Optional[Figure] = None,
    row: int = 1,
    col: int = 1,
    plotkwargs: Dict[str, Any] = {},
    layoutkwargs: Dict[str, Any] = {},
    **kwargs,
) -> Figure:
    """XY plot.

    - x-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``X``
    - y-axis : :math:`\\log_2{(\\text{gProcessedSignal})}` for each gene in sample ``Y``

    Args:
        df (pd.DataFrame)                          : DataFrame
        x (str)                                    : The column name for sample ``X``.
        y (str)                                    : The column name for sample ``Y``.
        logarithmic (bool)                         : Whether to log the values of ``df[x]`` and ``df[y]``
        color (Optional[str], optional)            : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        symbol (Optional[str], optional)           : The column name in ``df`` to assign symbols to marks. Defaults to ``None``.
        size (Optional[str], optional)             : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        hover_data (Optional[List[str]], optional) : Values in this column appear in bold in the hover tooltip. Defaults to ``None``.
        hover_name (Optional[str], optional)       : Values in this column appear in the hover tooltip. Defaults to ``None``.
        fig (Optional[Figure], optional)           : An instance of Figure.
        row (int, optional)                        : Row of subplots. Defaults to ``1``.
        col (int, optional)                        : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                 : Keyword arguments for ``go.Scatter``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])               : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>`. Defaults to ``{}``.

    Returns:
        Figure: An instance of ``Figure`` with XY plot.

    .. plotly::
        :include-source:
        :iframe-height: 600px

        >>> import pandas as pd
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
    fig = fig or subplots_create(nrows=1, ncols=1, style="plotly")
    df = df.copy(deep=True)
    if logarithmic:
        df[x] = df[x].apply(lambda x: np.log2(x))
        df[y] = df[y].apply(lambda y: np.log2(y))
    fig = trace_transition(
        from_fig=px.scatter(
            data_frame=df,
            x=x,
            y=y,
            color=color,
            symbol=symbol,
            size=size,
            hover_name=hover_name,
            hover_data=hover_data,
            **plotkwargs,
        ),
        to_fig=fig,
        row=row,
        col=col,
    )
    fig = update_layout(
        fig=fig,
        title=f"XY plot ({x} vs {y})",
        xlabel="$\\log_{2}(\\text{" + x + "})$",
        ylabel="$\\log_{2}(\\text{" + y + "})$",
        **layoutkwargs,
        **kwargs,
    )
    return fig


def MAplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    symbol: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    hover_name: Optional[str] = None,
    hlines: Union[Dict[Number, Dict[str, Any]], List[Number]] = [],
    fig: Optional[Figure] = None,
    row: int = 1,
    col: int = 1,
    plotkwargs: Dict[str, Any] = {},
    layoutkwargs: Dict[str, Any] = {},
    **kwargs,
) -> Figure:
    """MA plot.

    - x-axis (Average) : :math:`\\log_{10}{\\left(\\text{gProcessedSignal}_X\\times\\text{gProcessedSignal}_Y\\right)}`
    - y-axis (Minus)   : :math:`\\log_{2}{\\left(\\text{gProcessedSignal}_Y / \\text{gProcessedSignal}_X\\right)}`

    Args:
        df (pd.DataFrame)                                       : DataFrame
        x (str)                                                 : The column name for sample ``X``.
        y (str)                                                 : The column name for sample ``Y``.
        color (Optional[str], optional)                         : The column name in ``df`` to assign color to marks. Defaults to ``None``.
        symbol (Optional[str], optional)                        : The column name in ``df`` to assign symbols to marks. Defaults to ``None``.
        size (Optional[str], optional)                          : The column name in ``df`` to assign mark sizes. Defaults to ``None``.
        hover_data (Optional[List[str]], optional)              : Values in this column appear in bold in the hover tooltip. Defaults to ``None``.
        hover_name (Optional[str], optional)                    : Values in this column appear in the hover tooltip. Defaults to ``None``.
        hlines (Union[Dict[Number,Dict[str,Any]],List[Number]]) : Height (``y``) to draw a horizon. If given a dictionary, values means kwargs of ``ax.hlines``
        fig (Optional[Figure], optional)                        : An instance of Figure.
        row (int, optional)                                     : Row of subplots. Defaults to ``1``.
        col (int, optional)                                     : Column of subplots. Defaults to ``1``.
        plotkwargs (Dict[str,Any])                              : Keyword arguments for ``go.Scatter``. Defaults to ``{}``.
        layoutkwargs (Dict[str,Any])                            : Keyword arguments for :func:`update_layout <teilab.plot.plotly.update_layout>`. Defaults to ``{}``.

    Returns:
        Figure: An instance of ``Figure`` with MA plot.

    .. plotly::
        :include-source:
        :iframe-height: 600px

        >>> import pandas as pd
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
        >>> fig = MAplot(
        ...     df=df_combined,
        ...     x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], hover_name="SystematicName",
        ...     hlines={
        ...         -1 : dict(fillcolor="red", marker={"color":"red"}, line={"width":1}, showlegend=False),
        ...         0  : dict(fillcolor="red", marker={"color":"red"}, line={"width":3}, showlegend=False),
        ...         1  : dict(fillcolor="red", marker={"color":"red"}, line={"width":1}, showlegend=False),
        ...     },
        ...     height=600, width=600,
        >>> )
        >>> fig.show()
    """
    fig = fig or subplots_create(nrows=1, ncols=1, style="plotly")
    df = df.copy(deep=True)
    x_axis_colname = "x-axis"
    y_axis_colname = "y-axis"
    X = np.log10(df[x] * df[y])
    Y = np.log2(df[y] / df[x])
    df[x_axis_colname] = X
    df[y_axis_colname] = Y
    fig = trace_transition(
        from_fig=px.scatter(
            data_frame=df,
            x=x_axis_colname,
            y=y_axis_colname,
            color=color,
            symbol=symbol,
            size=size,
            hover_name=hover_name,
            hover_data=hover_data,
            **plotkwargs,
        ),
        to_fig=fig,
        row=row,
        col=col,
    )
    if isinstance(hlines, list):
        hlines = dict(zip(hlines, [{} for _ in range(len(hlines))]))
    for key, hlineskw in hlines.items():
        fig.add_trace(
            go.Scatter(x=[min(X), max(X)], y=[key, key], legendgroup=f"{col}-{row}", **hlineskw), row=row, col=col
        )
    fig = update_layout(
        fig=fig,
        title=f"MA plot ({x} vs {y})",
        xlabel="$\\log_{10}(\\text{" + x + "}\\times\\text{" + y + "})$",
        ylabel="$\\log_{2}(\\left(\\text{" + y + "} / \\text{" + x + "}\\right))$",
        **layoutkwargs,
        **kwargs,
    )
    return fig


def update_layout(
    fig: Figure,
    row: int = 1,
    col: int = 1,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[Any, Any]] = None,
    ylim: Optional[Tuple[Any, Any]] = None,
    xrangeslider: Dict[str, Any] = {"visible": False},
    font: Dict[str, Any] = {"family": "verdana, arial, sans-serif", "size": 14},
    legend: bool = True,
    xaxis_type: str = "linear",
    yaxis_type: str = "linear",
    legend_tracegroupgap: int = 100,
    width: int = 600,
    height: int = 600,
    template: str = "presentation",
) -> Figure:
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
        legend_tracegroupgap (int, optional)      : Gap between legend groups. Defaults to ``100``.
        width (int, optional)                     : Figure width. Defaults to ``600``.
        height (int, optional)                    : Figure height. Defaults to ``600``.
        template (str, optional)                  : Plotly themes. You can check the all template by ``python -c "import plotly.io as pio; print(pio.templates)``. Defaults to ``"plotly_white"``.

    Returns:
        Figure: Figure object with layout updated.

    .. plotly::
        :include-source:
        :iframe-height: 400px

        >>> import plotly.graph_objects as go
        >>> from teilab.utils import subplots_create
        >>> from teilab.plot.plotly import update_layout
        >>> fig = subplots_create(nrows=1, ncols=2, style="plotly")
        >>> for c in range(1,3):
        ...     fig.add_trace(go.Scatter(x=[1,2,3],y=[4,5,6]),row=1,col=c)
        >>> fig = update_layout(fig=fig, title="Sample", ylim=(4.5,5.5), col=2, height=400)
        >>> fig.show()
    """
    fig.update_xaxes(title=xlabel, range=xlim, rangeslider=xrangeslider, row=row, col=col)
    fig.update_yaxes(title=ylabel, range=ylim, row=row, col=col)
    fig.update_layout(
        title=title,
        font=font,
        width=width,
        height=height,
        template=template,
        showlegend=legend,
        xaxis_type=xaxis_type,
        yaxis_type=yaxis_type,
        legend_tracegroupgap=legend_tracegroupgap,
    )
    return fig
