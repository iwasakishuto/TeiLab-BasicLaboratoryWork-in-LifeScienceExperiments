import numpy as np
from teilab.utils import dict2str, subplots_create
from teilab.plot.plotly import density_plot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
nfigs = len(kwargses)
fig = subplots_create(ncols=nfigs, style="plotly")
for i,(kwargs) in enumerate(kwargses, start=1):
    _ = density_plot(data, fig=fig, title=title, col=i, legend=False, width=1000, height=400, **kwargs)
fig.show()
