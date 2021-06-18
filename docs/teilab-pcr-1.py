import numpy as np
from teilab.pcr import calibration_curve_plot
from teilab.utils import subplots_create
qualities = np.asarray([2, 1, 0.1])
calibrations = np.asarray([
    [[21.9904747,  21.94359016], [23.2647419,  23.24508476], [27.37600517, 27.46136856]],
    [[26.68031502, 26.75434494], [28.25722122, 28.239748  ], [31.77442169, 32.42930222]]
])
targets   = ["Gapdh", "VIM"]
fig, axes = subplots_create(ncols=2, style="matplotlib", figsize=(12,4))
for ax,cts,target in zip(axes, calibrations, targets):
    _ = calibration_curve_plot(qualities, cts, ax=ax, target=target)
fig.show()
