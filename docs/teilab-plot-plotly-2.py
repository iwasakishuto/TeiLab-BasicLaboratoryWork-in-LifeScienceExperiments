import numpy as np
from teilab.utils import dict2str, subplots_create
from teilab.plot.plotly import boxplot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples, n_features))
kwargses = [{"vert":True},{"vert":False}]
title = ", ".join([dict2str(kwargs) for kwargs in kwargses])
nfigs = len(kwargses)
fig = subplots_create(ncols=nfigs, style="plotly")
for col,kwargs in enumerate(kwargses, start=1):
    _ = boxplot(data, title=title, fig=fig, col=col, width=1000, height=400, **kwargs)
fig.show()
