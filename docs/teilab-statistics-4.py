import numpy as np
from teilab.utils import subplots_create
from teilab.statistics import paired_t_test
fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
A = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
B = np.array([1.9,  0.8,  1.1,  0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
for ax,alternative in zip(axes,["less","two-sided","greater"]):
    paired_t_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
fig.show()
