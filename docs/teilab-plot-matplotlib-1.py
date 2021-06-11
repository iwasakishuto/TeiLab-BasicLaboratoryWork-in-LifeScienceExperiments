import numpy as np
from teilab.utils import dict2str, subplots_create
from teilab.plot.matplotlib import density_plot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
nfigs = len(kwargses)
fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
for ax,kwargs in zip(axes,kwargses):
    _ = density_plot(data, ax=ax, title=dict2str(kwargs), **kwargs)
fig.show()
