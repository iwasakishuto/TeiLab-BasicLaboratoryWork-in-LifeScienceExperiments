#coding: utf-8
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from matplotlib.colors import Colormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mplFigure
from plotly.graph_objs import Figure as plotlyFigure
from matplotlib.axes import Axes
from typing import Any,Dict,List,Tuple,Optional,Union

def get_colorList(n:int, cmap:Optional[Union[str,Colormap]]=None, style:str="matplotlib") -> List[Tuple[float,float,float,float]]:
    """Get a color List using matplotlib's colormaps. See `Choosing Colormaps in Matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>` for details.

    Args:
        n (int)                                        : The number of samples
        cmap (Optional[Union[str,Colormap]], optional) : A ``Colormap`` object or a color map name. Defaults to ``None``.
        style (str)                                    : How to express colors (Please choose from ``"matplotlib"``, or ``"plotly"``)

    Returns:
        List[Tuple[float,float,float,float]]: Color List

    Examples:
        >>> import matplotlib
        >>> from matplotlib.cm import _cmap_registry
        >>> from teilab.utils import get_colorList
        >>> get_colorList(n=3, cmap="bwr")
        [(0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)]
        >>> get_colorList(n=3, cmap=_cmap_registry["bwr"])
        [(0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)]
        >>> get_colorList(n=3)
        [(0.190631, 0.407061, 0.556089, 1.0),
        (0.20803, 0.718701, 0.472873, 1.0),
        (0.993248, 0.906157, 0.143936, 1.0)]
        >>> matplotlib.rcParams['image.cmap'] = "bwr"
        >>> get_colorList(n=3)
        [(0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)]
        >>> get_colorList(n=3, cmap="bwr", style="plotly")
        ['rgba(170,170,255,1.0)', 'rgba(255,170,170,1.0)', 'rgba(255,0,0,1.0)']
    """
    cmap = plt.get_cmap(name=cmap)
    colors = [cmap((i+1)/n) for i in range(n)]
    if style in ["plotly", "rgba"]:
        colors = [f'rgba({",".join([str(int(e*255)) if i<3 else str(e) for i,e in enumerate(color)])})' for color in colors]
    return colors

def subplots_create(nrows:int=1, ncols:int=1, sharex:Union[bool,str]=False, sharey:Union[bool,str]=False,
                    style:str="matplotlib", **kwargs) -> Union[Tuple[mplFigure,Axes],plotlyFigure]:
    """Create subplots for each plot style.

    Args:
        nrows (int, optional)              : Number of rows of the subplot grid. Defaults to ``1``.
        ncols (int, optional)              : Number of columns of the subplot grid. Defaults to ``1``.
        sharex (Union[bool,str], optional) : Controls sharing of properties among x-axes. Defaults to ``False``.
        sharey (Union[bool,str], optional) : Controls sharing of properties among y-axes. Defaults to ``False``.
        style (str, optional)              : Plot style. Please choose from ``"matplotlib"``, or ``"plotly"`` . Defaults to ``"matplotlib"``.

    Returns:
        Union[Tuple[mplFigure,Axes],plotlyFigure]: Subplots to suit each plot style.

    Examples:
        >>> from teilab.utils import subplots_create
        >>> fig,axes = subplots_create(nrows=3, style="matplotlib")
        >>> fig.__class__
        >>> "<class 'matplotlib.figure.Figure'>"
        >>> str(axes[0].__class__)
        >>> "<class 'matplotlib.axes._subplots.AxesSubplot'>"
        >>> fig = subplots_create(nrows=3, style="plotly")
        >>> str(fig.__class__)
        >>> "<class 'plotly.graph_objs._figure.Figure'>"
    """
    if style == "plotly":
        return make_subplots(rows=nrows, cols=ncols, shared_xaxes=sharex, shared_yaxes=sharey, **kwargs)
    else:
        return plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, **kwargs)