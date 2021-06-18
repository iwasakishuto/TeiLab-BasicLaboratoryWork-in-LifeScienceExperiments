import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from teilab.pcr import expression_ratio_plot, calibration_curve_plot
fig = plt.figure(constrained_layout=True, figsize=(14,6))
gs = GridSpec(nrows=2, ncols=4, figure=fig)
targets   = ["Gapdh", "VIM"]
qualities = np.asarray([2, 1, 0.1])
calibration_cts = np.asarray([
  [[21.9904747,  21.94359016], [23.2647419,  23.24508476], [27.37600517, 27.46136856]],
  [[26.68031502, 26.75434494], [28.25722122, 28.239748  ], [31.77442169, 32.42930222]]
])
n_targets,n_qualities,n_trials_calibrations = calibration_cts.shape
print(
    "[Calibration]",
    f"{n_targets}_targets: {targets}",
    f"{n_qualities}_targets: {qualities}",
    f"n_trials : {n_trials_calibrations}",
    sep="\n"
)
efficiencies = []
for i,cts in enumerate(calibration_cts):
    ax, e = calibration_curve_plot(qualities, cts, ax=fig.add_subplot(gs[i,3]), target=targets[i])
    efficiencies.append(e)
samples = ["mock(1)", "mock(5)", "unmodified", "2OMe3", "2OMe5", "2OMe7", "LNA3", "LNA7"]
Cts = np.asarray([
    [[23.2647419, 23.24508476], [20.76102257, 20.77914238], [19.40455055, 19.52949905], [19.70094872, 19.60042572], [19.41954041, 19.13051605], [24.17935753, 21.98130798], [20.01245308, 20.02809715], [21.2081356, 20.1692791]],
    [[28.25722122, 28.239748], [25.16436958, 25.28390503], [24.71133995, 24.70510483], [25.37249184, 25.47054863], [24.72605515, 24.43961525], [27.91354942, 27.93320656], [26.08522797, 26.0483017], [24.96000481, 25.04871941]]
])
_,n_samples,n_trials_Cts = Cts.shape
print(
    "[Normalized Expression Ratio]",
    f"{n_samples}_samples: {samples}",
    f"n_trials : {n_trials_Cts}",
    sep="\n"
)
ax = expression_ratio_plot(GOI_Cts=Cts[1], REF_Cts=Cts[0], e_GOI=efficiencies[1], e_REF=efficiencies[0], name_GOI="VIM", name_REF="Gapdh", labels=samples, ax=fig.add_subplot(gs[:,:3]))
fig.show()
