import numpy as np
from teilab.utils import dict2str, subplots_create
from teilab.plot.plotly import cumulative_density_plot
n_samples, n_features = (4, 1000)
data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
fig = cumulative_density_plot(data, fig=None, xlabel="value", width=800, height=400)
fig.show()
