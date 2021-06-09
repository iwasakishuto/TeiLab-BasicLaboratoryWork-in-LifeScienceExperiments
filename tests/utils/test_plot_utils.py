# coding: utf-8
def test_get_colorList():
    import matplotlib
    from matplotlib.cm import _cmap_registry
    from teilab.utils import get_colorList
    assert get_colorList(n=3, cmap="bwr") == [
        (0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)
    ]
    assert get_colorList(n=3, cmap=_cmap_registry["bwr"]) == [
        (0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)
    ]
    assert get_colorList(n=3) == [
        (0.190631, 0.407061, 0.556089, 1.0),
        (0.20803, 0.718701, 0.472873, 1.0),
        (0.993248, 0.906157, 0.143936, 1.0)
    ]
    matplotlib.rcParams['image.cmap'] = "bwr"
    assert get_colorList(n=3) == [
        (0.6666666666666666, 0.6666666666666666, 1.0, 1.0),
        (1.0, 0.6666666666666667, 0.6666666666666667, 1.0),
        (1.0, 0.0, 0.0, 1.0)
    ]
    get_colorList(n=3, cmap="bwr", style="plotly") == ['rgba(170,170,255,1.0)', 'rgba(255,170,170,1.0)', 'rgba(255,0,0,1.0)']

def test_subplots_create():
    from teilab.utils import subplots_create
    fig,axes = subplots_create(nrows=3, style="matplotlib")
    assert str(fig.__class__) == "<class 'matplotlib.figure.Figure'>"
    assert str(axes[0].__class__) == "<class 'matplotlib.axes._subplots.AxesSubplot'>"
    fig = subplots_create(nrows=3, style="plotly")
    assert str(fig.__class__) == "<class 'plotly.graph_objs._figure.Figure'>"