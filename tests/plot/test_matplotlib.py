# coding: utf-8

def test_density_plot(db):
    from teilab.utils import dict2str, subplots_create
    from teilab.plot.matplotlib import density_plot
    data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
    kwarges = [{"bins":100},{"bins":10},{"bins":"auto"}]
    nfigs = len(kwarges)
    fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
    for ax,kwargs in zip(axes,kwarges):
        _ = density_plot(data, ax=ax, title=dict2str(kwargs), **kwargs)
    fig.show()

def test_cumulative_density_plot(db):
    from teilab.utils import dict2str, subplots_create
    from teilab.plot.matplotlib import cumulative_density_plot
    data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
    fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
    ax = cumulative_density_plot(data, ax=ax, xlabel="value")
    fig.show()

def test_boxplot(db):
    from teilab.utils import dict2str, subplots_create
    from teilab.plot.matplotlib import boxplot
    data = db.generate_normal_distributions(random_state=None, n_samples=4, n_features=1000)
    kwarges = [{"vert":True},{"vert":False}]
    nfigs = len(kwarges)
    fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
    for ax,kwargs in zip(axes,kwarges):
        _ = boxplot(data, title=dict2str(kwargs), ax=ax, **kwargs)
    fig.show()

def test_XYplot(db):
    from teilab.plot.matplotlib import XYplot
    from teilab.utils import subplots_create
    fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
    ax = XYplot(
        df=db.df_combined, 
        x=db.datasets.samples.Condition[0], 
        y=db.datasets.samples.Condition[1], 
        ax=ax
    )
    fig.show()

def test_MAplot(db):
    from teilab.plot.matplotlib import MAplot
    from teilab.utils import subplots_create
    fig, ax = subplots_create(figsize=(6,4), style="matplotlib")
    ax = MAplot(
        df=db.df_combined, 
        x=db.datasets.samples.Condition[0], 
        y=db.datasets.samples.Condition[1], 
        ax=ax
    )
    fig.show()        