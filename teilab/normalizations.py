#coding: utf-8
"""

Differential gene expression can be an outcome of true biological variability or experimental artifacts. Normalization techniques have been used to minimize the effect of experimental artifacts on differential gene expression analysis.

###############################
Robust Multichip Analysis (RMA)
###############################

In microarray analysis, many algorithms have been proposed, but the most widely used one (**de facto standard**) is :fa:`file-pdf-o` `Robust Multichip Analysis (RMA) <https://academic.oup.com/biostatistics/article/4/2/249/245074>`_ , where the signal value of each spot ( ``RawData`` ) is processed and normalized according to the following flow. ( :ref:`1) Background Subtraction <target to background subtraction section>`, :ref:`2) Normalization Between Samples <target to normalization between samples section>` and :ref:`3) Summarization <target to summarization section>` )

.. graphviz:: _graphviz/RobustMultichipAnalysis.dot
      :class: popup-img

.. _target to background subtraction section:

*************************
1. Background Subtraction
*************************

バックグラウンド補正は、Non-specific Hybridization に由来するシグナル強度を差し引くためのものです。ここでは、「観察されたシグナルの強度は、真のシグナルの強度とバックグラウンドシグナルの強度とが合成されたものである」と仮定します。

そこで、あるマイクロアレイについて、

- 真のシグナル強度分布は指数分布
- バックグラウンドの強度分布は正規分布

となると仮定し、その仮定の元で最も現象を表現しているパラメータを推定することで、バックグラウンドの強度分布を計算し、これを差し引くことで真のシグナル強度を推定します。

TODO: もっとここを詰めねば。論文読みます。

.. _target to normalization between samples section:

********************************
2. Normalization Between Samples
********************************

ここでは、各マイクロアレイ「間の」正規化を行います。一つのマイクロアレイ実験の結果から言えることは非常に限られており、他の（処理群やコントロール群）実験結果と比較することが解析の基本になりますが、一般的に実験においては、実験操作や機材の特性によってバイアスが生じることは避けられません。

For example, if you want to characterize the changes in global gene expression in the livers of H1/siRNAinsulin-CMV/hIDE transgenic (Tg) mice in response to the reduced bioavailability of insulin [#ref1]_, and the expression level of each RNA in Tg mice was generally lower than that of non-Tg mice, **you may mistakenly conclude that almost all of the RNAs were down-regulated respectively by reduced bioavailability of insulin.**

.. [#ref1] :fa:`file-pdf-o` `Microarray analysis of insulin-regulated gene expression in the liver: the use of transgenic mice co-expressing insulin-siRNA and human IDE as an animal model <https://pubmed.ncbi.nlm.nih.gov/17982690/>`_

そこで、バイアスの影響が軽減するように、得られた値に補正をかけることが必要です。この操作は各手法ごとに定義されるある仮定に基づいているので、取り扱いには注意が必要です。

There are numerous proposals for normalizing unbalanced data between samples (:fa:`file-pdf-o` `This review paper <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6988798/>`_ summarizes 23 normalization methods developed for unbalanced transcriptome data), but we will introduce only three of them.

.. note::

    1. :ref:`Percentile <target to percentile section>`
    2. :ref:`Quantile <target to quantile section>`

.. _target to percentile section:

1. Percentile
=============

This is a constant adjustoment in a global manner.

.. _target to quantile section:

2. Quantile
===========

- 遺伝子を発現量順に並べたとき、同じ順位の遺伝子は同じ発現量を示す。
- 各サンプルの遺伝子発現の強度分布はほとんど変わらない。


https://www.ncbi.nlm.nih.gov/pmc/articles/PMC100354/

.. _target to summarization section:

****************
3. Summarization
****************

https://github.com/scipy/scipy/blob/v1.6.3/scipy/signal/signaltools.py#L3384-L3467

"""
import numpy as np
from scipy.stats.mstats import gmean
from typing import Any
from nptyping import NDArray
from numbers import Number


def percentile(data:NDArray[(Any,Any),Number], percent:Number=75) -> NDArray[(Any,Any),Number]:
    """Perform Percentile Normalization.

    Args:
        data (NDArray[(Any,Any),Number]) : Input data. Shape = ( ``n_samples``, ``n_features`` )
        percent (Number, optional)       : Which percentile value to normalize. Defaults to ``75``.

    Returns:
        NDArray[(Any,Any),Number] : percentiled data. Shape = ( ``n_samples``, ``n_features`` )

    Raises:
        ValueError: When ``percent`` tiles contain negative values.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from teilab.normalizations import percentile
        >>> from teilab.plot.matplotlib import densityplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1),  size=(n_samples,n_features))
        >>> data_percentiled = percentile(data=data, percent=75)
        >>> fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,4))
        >>> ax = densityplot(data=data, title="Before Percentile", ax=axes[0])
        >>> ax = densityplot(data=data_percentiled, title="After Percentile", ax=axes[1])
    
    +--------------------------------------------------+
    |                      Results                     |
    +==================================================+
    | .. image:: _images/normalizations.percentile.jpg |
    |    :class: popup-img                             |
    +--------------------------------------------------+
    """    
    a = np.percentile(a=data, q=percent, axis=1)
    if np.any(a<0):
        for i,val in enumerate(a):
            if val<0: break
        raise ValueError(f"Geometric mean cannot be calculated because the {percent}%tiles contain negative values (ex. {i}-th data's {percent}%tile = {val} < 0) ")
    return data * np.expand_dims(gmean(a)/a, axis=1)

def quantile(data:NDArray[(Any,Any),Number]) -> NDArray[(Any,Any),Number]:
    """Perform Quantile Normalization.

    Args:
        data (NDArray[(Any,Any),Number]) : Input data. Shape = ( ``n_samples``, ``n_features`` )

    Returns:
        NDArray[(Any,Any),Number] : percentiled data. Shape = ( ``n_samples``, ``n_features`` )

    Raises:
        ValueError: When ``data`` contains negative values.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from teilab.normalizations import percentile
        >>> from teilab.plot.matplotlib import densityplot
        >>> n_samples, n_features = (4, 1000)
        >>> data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features), ) + 3.5
        >>> data_quantiled = quantile(data=data)
        >>> fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,4))
        >>> ax = densityplot(data=data, title="Before Quantile", ax=axes[0])
        >>> ax = densityplot(data=data_quantiled, title="After Quantile", ax=axes[1])

    +------------------------------------------------+
    |                      Results                   |
    +================================================+
    | .. image:: _images/normalizations.quantile.jpg |
    |    :class: popup-img                           |
    +------------------------------------------------+
    """
    if np.any(data<0):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i,j] < 0:
                    break
        raise ValueError(f"Geometric mean cannot be calculated because ``data`` contain negative values. (ex. data[{i}][{j}] = {data[i,j]} < 0)")
    return gmean(a=np.sort(a=data, axis=1), axis=0)[np.argsort(np.argsort(data, axis=1), axis=1)]

def lowess(data:NDArray[(Any,Any),Number]) -> NDArray[(Any,Any),Number]:
    """Perform LOWESS Normalization.

    Args:
        data (NDArray[(Any,Any),Number]) : Input data. Shape = ( ``n_samples``, ``n_features`` )

    Returns:
        NDArray[(Any,Any),Number] : percentiled data. Shape = ( ``n_samples``, ``n_features`` )
    """
    return data

# def normalization_between_samples(df:pd.DataFrame):
