# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


class PCRresult:
    def __init__(self):
        self.samples = ["mock(1)", "mock(5)", "unmodified", "2OMe3", "2OMe5", "2OMe7", "LNA3", "LNA7"]
        self.REF_name = "Gapdh"
        self.GOI_name = "VIM"
        self.qualities = np.asarray([2, 1, 0.1])
        self.REF_calibration_Cts = np.asarray(
            [
                [21.9904747, 21.94359016],
                [23.2647419, 23.24508476],
                [27.37600517, 27.46136856],
            ]
        )
        self.GOI_calibration_Cts = np.asarray(
            [[26.68031502, 26.75434494], [28.25722122, 28.239748], [31.77442169, 32.42930222]]
        )
        self.REF_Cts = np.asarray(
            [
                [23.2647419, 23.24508476],
                [20.76102257, 20.77914238],
                [19.40455055, 19.52949905],
                [19.70094872, 19.60042572],
                [19.41954041, 19.13051605],
                [24.17935753, 21.98130798],
                [20.01245308, 20.02809715],
                [21.2081356, 20.1692791],
            ]
        )
        self.GOI_Cts = np.asarray(
            [
                [28.25722122, 28.239748],
                [25.16436958, 25.28390503],
                [24.71133995, 24.70510483],
                [25.37249184, 25.47054863],
                [24.72605515, 24.43961525],
                [27.91354942, 27.93320656],
                [26.08522797, 26.0483017],
                [24.96000481, 25.04871941],
            ]
        )


def test_pcr_analysis():
    pcr_result = PCRresult()
    from teilab.pcr import calibration_curve_plot, expression_ratio_plot

    fig = plt.figure(constrained_layout=True, figsize=(14, 6))
    gs = GridSpec(nrows=2, ncols=4, figure=fig)
    _, e_REF = calibration_curve_plot(
        qualities=pcr_result.qualities,
        cts=pcr_result.REF_calibration_Cts,
        ax=fig.add_subplot(gs[0, 3]),
        target=pcr_result.REF_name,
    )
    _, e_GOI = calibration_curve_plot(
        qualities=pcr_result.qualities,
        cts=pcr_result.GOI_calibration_Cts,
        ax=fig.add_subplot(gs[1, 3]),
        target=pcr_result.GOI_name,
    )
    ax = expression_ratio_plot(
        GOI_Cts=pcr_result.GOI_Cts,
        REF_Cts=pcr_result.REF_Cts,
        e_GOI=e_GOI,
        e_REF=e_REF,
        name_GOI=pcr_result.GOI_name,
        name_REF=pcr_result.REF_name,
        labels=pcr_result.samples,
        ax=fig.add_subplot(gs[:, :3]),
    )
    fig.show()
