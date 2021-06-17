import numpy as np
from teilab.utils import subplots_create
from teilab.statistics import wilcoxon_test
fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
rnd = np.random.RandomState(0)
A,B = rnd.random_sample(size=(2,30))
for ax,alternative in zip(axes,["less","two-sided","greater"]):
    wilcoxon_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
fig.show()
