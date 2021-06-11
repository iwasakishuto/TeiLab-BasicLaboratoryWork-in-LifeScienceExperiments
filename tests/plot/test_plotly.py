# coding: utf-8

def test_density_plot(db):
    from teilab.utils import dict2str, subplots_create
    from teilab.plot.plotly import density_plot
    data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
    kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
    title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
    nfigs = len(kwargses)
    fig = subplots_create(ncols=nfigs, style="plotly")
    for i,kwargs in enumerate(kwargses, start=1):
        _ = density_plot(data, fig=fig, title=title, col=i, legend=False, width=1000, height=400, **kwargs)
    # fig.show()

def test_cumulative_density_plot(db):
    from teilab.utils import dict2str, subplots_create
    from teilab.plot.plotly import cumulative_density_plot
    data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
    fig = cumulative_density_plot(data, fig=None, xlabel="value", width=800, height=400)
    # fig.show()

# def test_boxplot(db):
#     from teilab.utils import dict2str, subplots_create
#     from teilab.plot.plotly import boxplot
#     data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
#     kwarges = [{"vert":True},{"vert":False}]
#     nfigs = len(kwarges)
#     fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
#     for ax,kwargs in zip(axes,kwarges):
#         _ = boxplot(data, title=dict2str(kwargs), ax=ax, **kwargs)
#     fig.show()

def test_XYplot(db):
    from teilab.plot.plotly import XYplot
    fig = XYplot(
        df=db.df_combined, 
        x=db.datasets.samples.Condition[0], 
        y=db.datasets.samples.Condition[1], 
        fig=None, height=600, width=600,
    )
    # fig.show()

def test_MAplot(db):
    from teilab.plot.plotly import MAplot
    fig = MAplot(
        df=db.df_combined, 
        x=db.datasets.samples.Condition[0], 
        y=db.datasets.samples.Condition[1], 
        fig=None, height=600, width=600,
    )
    # fig.show()        