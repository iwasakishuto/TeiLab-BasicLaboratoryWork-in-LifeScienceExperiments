import matplotlib.pyplot as plt
from teilab.utils import dict2str, subplots_create
from teilab.plot.plotly import densityplot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
kwargses = [{"bins":100},{"bins":10},{"bins":"auto"}]
title = ", ".join([f"{i}:{dict2str(kwargs)}" for i,kwargs in enumerate(kwargses)])
nfigs = len(kwargses)
fig = subplots_create(ncols=nfigs, style="plotly")
for i,(kwargs) in enumerate(kwargses, start=1):
    _ = densityplot(data, fig=fig, title=title, col=i, legend=False, width=1000, height=400, **kwargs)
fig.show()
