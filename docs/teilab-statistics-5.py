import numpy as np
from teilab.utils import subplots_create
from teilab.statistics import mann_whitney_u_test
fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
A = np.array([1.83, 1.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 2.01, 3.11])
B = np.array([0.88, 0.65, 0.60, 1.05, 1.06, 1.29, 1.06, 2.14, 1.29])
for ax,alternative in zip(axes,["less","two-sided","greater"]):
    mann_whitney_u_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
fig.show()
