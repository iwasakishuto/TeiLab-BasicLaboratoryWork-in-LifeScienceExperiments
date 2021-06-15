import numpy as np
from teilab.utils import subplots_create
from teilab.statistics import welch_t_test
fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
B = np.array([4.8, 2.1, 5.1,  2.0, 4.0,  1.0,  3.4,  2.7,  5.1, 1.4, 1.6])
for ax,alternative in zip(axes,["less","two-sided","greater"]):
    welch_t_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
fig.show()
