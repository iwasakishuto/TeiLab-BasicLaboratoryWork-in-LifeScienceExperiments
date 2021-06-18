#coding: utf-8
r"""
A **real-time polymerase chain reaction (real-time PCR)** is a laboratory technique of molecular biology based on the **polymerase chain reaction (PCR)**. It monitors the amplification of a targeted DNA molecule **during** the PCR (i.e., in real time), **not at its end**, as in conventional PCR. Real-time PCR can be used quantitatively (quantitative real-time PCR) and semi-quantitatively (i.e., above/below a certain amount of DNA molecules) (semi-quantitative real-time PCR).

############
Instructions
############

There are two basic quantification methods in real-time PCR, :ref:`"Absolute Quantification" <target to Absolute Quantification section>` and :ref:`"Relative Quantification" <target to Relative Quantification section>` , but the underlying ideas are the same. As the template DNA increases :math:`2 (1 + e (\text{amplification efficiency}))` times in one cycle of PCR, you can calculate the initial amount of DNA by detecting how many cycles of PCR were performed to reach a certain amount of DNA.

.. _target to Absolute Quantification section:

***********************
Absolute Quantification
***********************

The standard curve method for **absolute quantification** is similar to that for **relative quantification**, except the absolute quantities of the standards must first be known by some independent means.

Create several dilution series from a known concentration target template and create an **accurate** calibration curve that associates each target concentration with the :math:`Ct` value. 

From this calibration curve, we can determine the actual copy number of the target DNAs using :math:`Ct` value.

.. _target to Relative Quantification section:

***********************
Relative Quantification
***********************

A method to focus on the **relative ratio** of a calibrator (control) sample and the gene of interest (GOI). The following various calibration algorithms have been proposed.

1. :ref:`ΔCt method <target to delta Ct method>`
2. :ref:`ΔΔCt method <target to delta delta Ct method>`
3. :ref:`Calibration curve method <target to Calibration curve method>`

.. _target to delta Ct method:

1. ΔCt method
==============

.. math::
    \text{Expression Ratio} = 2^{-\Delta Ct_{GOI}}\quad\left(GOI: Gene of Interest.\right)

.. math::
    \begin{cases}
        \Delta Ct_{GOI}&=Ct_{GOI}^{sample}-Ct_{GOI}^{cont.},\\
        ratio &= \left(\displaystyle\frac{[DNA]_0^{sample}}{[DNA]_0^{cont.}}\right)^{\displaystyle\frac{1}{\log_2(1+e)}}.
    \end{cases}

.. _target to delta delta Ct method:

2. ΔΔCt method
===============

.. math::
    \text{Expression Ratio} = 2^{-\Delta\Delta Ct_{GOI}}.

It is necessary that the amplification efficiencies of REF (Reference Gene) and GOI (Gene of Interest) are the same, but this assumption is generally not met.

.. math::
    \Delta\Delta Ct = \Delta Ct_{GOI} - \Delta Ct_{REF}.

.. _target to Calibration curve method:

3. Calibration curve method
===========================

.. math::
    \text{Expression Ratio} = \left(E_{GOI}^{-\Delta Ct_{GOI}}\right) / \left(E_{REF}^{-\Delta Ct_{REF}}\right)

A calibration curve is drawn to determine the amplification efficiency of **both REF and GOI**, and this is used to standardize the difference in :math:`Ct` of GOI contained in the target sample and the control sample with respect to REF. The amount of REF contained in the sample must be equal for all samples.

.. math::
    E =  10^{[-1/\text{slope}]}\left(=1+e\right)\quad(\text{efficiency from calibration curve.})

Let Amount (concentration) of initial template DNA be :math:`[DNA]_0`, amplification efficiency be :math:`e`, and Number of PCR cycles be :math:`C`.

The relationship between the amount of DNA (:math:`[DNA]`) in the PCR product amplified by PCR and the number of cycles (:math:`C`) is shown below:

.. math::
    [DNA] &= [DNA]_0\left(1 + e\right)^C\\ C &= \frac{\log[DNA]-\log[DNA]_0}{\log(1+e)}

Therefore, let the number of cycles :math:`C` that reaches the **threshold value** be :math:`Ct`, and the DNA concentration at that time is :math:`[DNA]_t`, the logarithm initial concentration is proportional to :math:`Ct` when the primers (:math:`\fallingdotseq e`) are the same, so this relationship can be represented by a calibration curve (**linear line**), and its slope (:math:`\text{slope}`) is shown below:

.. math::
    \text{slope} 
    &= \frac{Ct^1-Ct^2}{\log[DNA]_0^1 - \log[DNA]_0^2}\\
    &= \frac{\left(\log[DNA]_0^2 - \log[DNA]_0^1\right) / \log(1+e)}{\log[DNA]_0^1 - \log[DNA]_0^2} \\
    &= \frac{-1}{\log\left(1+e\right)}\\
    \therefore e &= 10^{\frac{-1}{\text{slope}}} - 1

Here, let :math:`E = 10^{\frac{-1}{slope}} = 1+e`, it can be written as follows

.. math::
    E_{GOI}^{-\Delta Ct_{GOI}} 
    &= \left(1 + e_{GOI}\right)^{\displaystyle\frac{\log[DNA]_{0,GOI}^{sample}-\log[DNA]_{0,GOI}^{cont.}}{\log\left(1+e_{GOI}\right)}}\\
    &= \frac{[DNA]_{0,GOI}^{sample}}{[DNA]_{0,GOI}^{cont.}}

so, the expression level of the calibrator (control) sample and the target sample can be compared in a **primer-independent manner**.

Therefore, by comparing this value with REF (ex. Housekeeping gene), it is possible to correctly compare the amount of DNA between different samples.

##############
Python Objects
##############
"""
import numpy as np

from typing import Any,Tuple,Optional,List
from matplotlib.axes import Axes
from nptyping import NDArray

from .utils.math_utils import optimize_linear
from .utils.plot_utils import get_colorList, subplots_create
from .plot.matplotlib import update_layout

def calibration_curve_plot(qualities:NDArray[Any,float], cts:NDArray[(Any,Any,Any),float], target:str="", color:str="#c94663", ecolor:str="black", ax:Optional[Axes]=None, **kwargs) -> Tuple[Axes,float]:
    """Plot a calibration curve.

    Args:
        qualities (NDArray[Any,float]) : Concentration of dilution series. shape=(n_qualities)
        cts (NDArray[(Any,Any),float]) : Ct values for calibration. shape=(n_qualities, n_trials_calibrations) 
        target (str, optional)         : The calibration target. Defaults to ``"Calibration curve"``.
        color (str, optional)          : The color of plot. Defaults to ``"#c94663"``.
        ecolor (str, optional)         : The color of Error Bar. Defaults to ``"#c94663"``.
        ax (Optional[Axes], optional)  : An instance of ``Axes``. Defaults to ``None``.

    Returns:
        Tuple[Axes,float]: An instance of ``Axes`` with calibration curve, and amplification efficiency.

    Raises:
        ValueError: ``cts.shape[0]`` is not the same as ``len(qualities)``.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.pcr import calibration_curve_plot
        >>> from teilab.utils import subplots_create
        >>> qualities = np.asarray([2, 1, 0.1])
        >>> calibrations = np.asarray([
        ...     [[21.9904747,  21.94359016], [23.2647419,  23.24508476], [27.37600517, 27.46136856]],
        ...     [[26.68031502, 26.75434494], [28.25722122, 28.239748  ], [31.77442169, 32.42930222]]
        >>> ])
        >>> targets   = ["Gapdh", "VIM"]
        >>> fig, axes = subplots_create(ncols=2, style="matplotlib", figsize=(12,4))
        >>> for ax,cts,target in zip(axes, calibrations, targets):
        ...     _ = calibration_curve_plot(qualities, cts, ax=ax, target=target)
        >>> fig.show()
    """
    n_qualities,n_trials_calibrations = cts.shape
    if len(qualities)!=n_qualities:
        raise TypeError(f"cts.shape[0] must be the same as the length of qualities, but got ({cts.sape[0]}!={len(qualities)})")
    log_qualities = np.log10(qualities)
    cts_mean = np.mean(cts.reshape(n_qualities,-1), axis=1)
    cts_std = np.std(cts.reshape(n_qualities,-1), axis=1)
    slope, intercept, func = optimize_linear(X=log_qualities, Y=cts_mean)
    e = 10**(-1/slope)-1
    
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib")[1]
    X = np.linspace(np.min(log_qualities), np.max(log_qualities), 1000)
    ax.plot(X, func(X), color=color)
    ax.errorbar(x=log_qualities, y=cts_mean, yerr=cts_std, capsize=5, fmt="o", ecolor=ecolor, color=color)
    ax = update_layout(
        ax=ax, title=f"Calibration curve for '{target}' (e={e:.3f})",
        xlabel="Relative initial concentration", ylabel="Cts", 
        **kwargs
    )
    ax.set_xticks(log_qualities)
    ax.set_xticklabels(qualities)
    return (ax, e)

def expression_ratio_plot(GOI_Cts:NDArray[(Any,Any),float], REF_Cts:NDArray[(Any,Any),float], e_GOI:float, e_REF:float, labels:List[str], name_GOI:str="", name_REF:str="", color:str="#c94663", ax:Optional[Axes]=None, **kwargs) -> Axes:
    r"""Plot to compare Expression Ratios.

    Args:
        GOI_Cts, REF_Cts (NDArray[(Any,Any),float]) : Threshold Cycles of GOI or REF. shape=(n_samples, n_trials_Cts)
        e_GOI, e_REF (float)                        : Efficiency of GOI or REF primer.
        labels (List[str])                          : [description] 
        name_GOI, name_REF (str, optional)          : The name of GOI or REF. Defaults to ``""``.
        color (str, optional)                       : The color of plot. Defaults to ``"#c94663"``.
        ax (Optional[Axes], optional)               : An instance of ``Axes``. Defaults to ``None``.

    Returns:
        Axes: An instance of ``Axes`` with expression ratio plots.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.gridspec import GridSpec
        >>> from teilab.pcr import expression_ratio_plot, calibration_curve_plot
        >>> fig = plt.figure(constrained_layout=True, figsize=(14,6))
        >>> gs = GridSpec(nrows=2, ncols=4, figure=fig)
        >>> targets   = ["Gapdh", "VIM"]
        >>> qualities = np.asarray([2, 1, 0.1])
        >>> calibration_cts = np.asarray([
        ...   [[21.9904747,  21.94359016], [23.2647419,  23.24508476], [27.37600517, 27.46136856]],
        ...   [[26.68031502, 26.75434494], [28.25722122, 28.239748  ], [31.77442169, 32.42930222]]
        >>> ])
        >>> n_targets,n_qualities,n_trials_calibrations = calibration_cts.shape
        >>> print(
        ...     "[Calibration]",
        ...     f"{n_targets}_targets: {targets}",
        ...     f"{n_qualities}_targets: {qualities}",
        ...     f"n_trials : {n_trials_calibrations}",
        ...     sep="\n"
        >>> )
        >>> efficiencies = []
        >>> for i,cts in enumerate(calibration_cts):
        ...     ax, e = calibration_curve_plot(qualities, cts, ax=fig.add_subplot(gs[i,3]), target=targets[i])
        ...     efficiencies.append(e)
        >>> samples = ["mock(1)", "mock(5)", "unmodified", "2OMe3", "2OMe5", "2OMe7", "LNA3", "LNA7"]
        >>> Cts = np.asarray([
        ...     [[23.2647419, 23.24508476], [20.76102257, 20.77914238], [19.40455055, 19.52949905], [19.70094872, 19.60042572], [19.41954041, 19.13051605], [24.17935753, 21.98130798], [20.01245308, 20.02809715], [21.2081356, 20.1692791]],
        ...     [[28.25722122, 28.239748], [25.16436958, 25.28390503], [24.71133995, 24.70510483], [25.37249184, 25.47054863], [24.72605515, 24.43961525], [27.91354942, 27.93320656], [26.08522797, 26.0483017], [24.96000481, 25.04871941]]    
        >>> ])
        >>> _,n_samples,n_trials_Cts = Cts.shape
        >>> print(
        ...     "[Normalized Expression Ratio]",
        ...     f"{n_samples}_samples: {samples}",
        ...     f"n_trials : {n_trials_Cts}",
        ...     sep="\n"
        >>> )
        >>> ax = expression_ratio_plot(GOI_Cts=Cts[1], REF_Cts=Cts[0], e_GOI=efficiencies[1], e_REF=efficiencies[0], name_GOI="VIM", name_REF="Gapdh", labels=samples, ax=fig.add_subplot(gs[:,:3]))
        >>> fig.show()
    """
    n_samples, n_trials_Cts = GOI_Cts.shape
    if len(labels)!=n_samples: labels = [f"No.{i}" for i in range(n_samples)]
    GOI_mean = np.mean(GOI_Cts.reshape(n_samples,-1), axis=1)
    REF_mean = np.mean(REF_Cts.reshape(n_samples,-1), axis=1)
    ratios = ((e_GOI+1)**(-GOI_mean)) / ((e_REF+1)**(-REF_mean))
    ax = ax or subplots_create(ncols=1, nrows=1, style="matplotlib", figsize=(int(n_samples*1.6), int(n_samples*0.8)))[1]
    ax.bar(labels, ratios, color=color)
    update_layout(ax=ax, title=f"'{name_GOI}' Expression (Normalized by '{name_REF}')", xlabel="samples", ylabel="Relative Expression", **kwargs)
    ax.grid()
    return ax