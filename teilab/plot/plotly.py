#coding: utf-8
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
from typing import Any,Dict,List,Tuple,Optional

def densityplot(data, names=None, bins=100, range=None, fig=None, row=1, col=1, **kwargs):
    fig = fig or make_subplots(rows=1, cols=1)
    names = names or [f"data_{i+1}" for i in range(len(data))]
    for ith_data,name in zip(data,names):
        hist, bin_edges = np.histogram(a=ith_data, bins=bins, range=range, density=True)
        fig.add_trace(trace=go.Scatter(x=bin_edges[1:], y=hist, name=name, mode="lines"), row=row, col=col)
    fig = update_layout(fig, row=row, col=col, **kwargs)
    return fig

def XYplot(df:pd.DataFrame, x:str, y:str, 
           color:Optional[str]=None, symbol:Optional[str]=None, size:Optional[str]=None, 
           hover_data:Optional[List[str]]=None, hover_name:Optional[str]=None, 
           **kwargs) -> Figure:
    """XY plot.

    Args:
        df (pd.DataFrame): [description]
        x (str): [description]
        y (str): [description]
        color (Optional[str], optional) : [description]. Defaults to ``None``.
        symbol (Optional[str], optional) : [description]. Defaults to ``None``.
        size (Optional[str], optional) : [description]. Defaults to ``None``.
        hover_data (Optional[List[str]], optional) : [description]. Defaults to ``None``.
        hover_name (Optional[str], optional) : [description]. Defaults to ``None``.

    Returns:
        Figure: [description]
    """
    df_ = df.copy(deep=True)
    x_:str = x + "_"; y_:str = y + "_"
    df_[x_] = df[x].apply(lambda x:np.log2(x))
    df_[y_] = df[y].apply(lambda y:np.log2(y))
    fig = px.scatter(data_frame=df_, x=x_, y=y_, color=color, symbol=symbol, size=size, hover_name=hover_name, hover_data=hover_data)
    fig = update_layout(fig=fig, title=f"XY plot ({x} vs {y})", xlabel="$\log_{2}(\\text{" + x + "})$", ylabel="$\log_{2}(\\text{" + y + "})$", **kwargs)
    return fig

def MAplot(X, Y, color=None, hovertext=None, marker_size=3, fig=None, row=1, col=1, 
           xlabel="$\log_{10}XY$", ylabel="$\log_{2}(Y/X)$", title="MA plot", **kwargs):
    if fig is None:
        fig = make_subplots(rows=1, cols=1) 
    if color is None:
        color = np.zeros_like(X)
    x = np.log10(X*Y)
    y = np.log2(Y/X)
    for c in np.unique(color):
        idx = color==c
        fig.add_trace(
            trace=go.Scatter(
                x=x[idx], y=y[idx], mode='markers', name=str(c), hovertext=hovertext[idx],
                marker_color=c, marker_size=marker_size,
            ), row=row, col=col
        )
    fig = update_layout(fig, row=row, col=col, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
    return fig

def update_layout(fig:Figure, row:int=1, col:int=1, 
                  title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None, 
                  xlim:Optional[Tuple[Any,Any]]=None, ylim:Optional[Tuple[Any,Any]]=None, 
                  xrangeslider:Dict[str,Any]={"visible": False},
                  font:Dict[str,Any]={"family":"verdana, arial, sans-serif", "size": 14}, legend:bool=False,
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
        legend (bool, optional)                   : Whether to show legend. Defaults to ``False``.
        xaxis_type (str, optional)                : X axis type. Defaults to ``"linear"``.
        yaxis_type (str, optional)                : Y axis type. Defaults to ``"linear"``.
        width (int, optional)                     : Figure width. Defaults to ``600``.
        height (int, optional)                    : Figure height. Defaults to ``600``.
        template (str, optional)                  : Plotly themes. You can check the all template by ``python -c "import plotly.io as pio; print(pio.templates)``. Defaults to ``"plotly_white"``.

    Returns:
        Figure: Figure object with layout updated.

    Examples:
        >>> import  plotly.graph_objects as go
        >>> from plotly.subplots import make_subplots
        >>> from teilab.plot.plotly import update_layout
        >>> fig = make_subplots(rows=1, cols=1)
        >>> fig.add_trace(go.Scatter(x=[1,2,3],y=[4,5,6]), row=1,col=1)
        >>> fig.show()
        >>> fig = update_layout(fig=fig, title="Sample", ylim=(4.5,5.5))
        >>> fig.show()
    """
    fig.update_xaxes(title=xlabel, range=xlim, rangeslider=xrangeslider, row=row, col=col)
    fig.update_yaxes(title=ylabel, range=ylim, row=row, col=col)
    fig.update_layout(
        title=title, font=font, width=width, height=height, template=template,
        showlegend=legend, xaxis_type=xaxis_type, yaxis_type=yaxis_type,
    )
    return fig
