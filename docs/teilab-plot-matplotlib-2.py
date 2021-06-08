from teilab.utils import dict2str, subplots_create
from teilab.plot.matplotlib import boxplot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
kwarges = [{"vert":True},{"vert":False}]
nfigs = len(kwarges)
fig, axes = subplots_create(ncols=nfigs, figsize=(int(6*nfigs),4), style="matplotlib")
for ax,kwargs in zip(axes,kwarges):
    _ = boxplot(data, title=dict2str(kwargs), ax=ax, **kwargs)
fig.show()
