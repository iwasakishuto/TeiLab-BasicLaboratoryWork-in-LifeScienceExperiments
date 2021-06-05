#coding: utf-8
"""

Differential gene expression can be an outcome of true biological variability or experimental artifacts. Normalization techniques have been used to minimize the effect of experimental artifacts on differential gene expression analysis.

###############################
Robust Multichip Analysis (RMA)
###############################

In microarray analysis, many algorithms have been proposed, but the most widely used one is :fa:`file-pdf-o` `Robust Multichip Analysis (RMA) <https://academic.oup.com/biostatistics/article/4/2/249/245074>`_ , where the signal value of each spot ( ``RawData`` ) is processed and normalized according to the following flow.

.. graphviz:: _graphviz/RobustMultichipAnalysis.dot

*************************
1. Background Subtraction
*************************

バックグラウンド補正は、Non-specific Hybridization に由来するシグナル強度を差し引くためのものです。ここでは、「観察されたシグナルの強度は、真のシグナルの強度とバックグラウンドシグナルの強度とが合成されたものである」と仮定します。

そこで、あるマイクロアレイについて、

- 真のシグナル強度分布は指数分布
- バックグラウンドの強度分布は正規分布

となると仮定し、その仮定の元で最も現象を表現しているパラメータを推定することで、バックグラウンドの強度分布を計算し、これを差し引くことで真のシグナル強度を推定します。

TODO: もっとここを詰めねば。論文読みます。


********************************
2. Normalization Between Samples
********************************

ここでは、各マイクロアレイ「間の」正規化を行います。一つのマイクロアレイ実験の結果から言えることは非常に限られており、他の（処理群やコントロール群）実験結果と比較することが解析の基本になりますが、一般的に実験においては、実験操作や機材の特性によってバイアスが生じることは避けられません。

For example, if you want to :fa:`file-pdf-o` `characterize the changes in global gene expression in the livers of H1/siRNAinsulin-CMV/hIDE transgenic (Tg) mice in response to the reduced bioavailability of insulin <https://pubmed.ncbi.nlm.nih.gov/17982690/>`_ , and the expression level of each RNA in Tg mice was generally lower than that of non-Tg mice, **you may mistakenly conclude that almost all of the RNAs were down-regulated respectively by reduced bioavailability of insulin.**

そこで、バイアスの影響が軽減するように、得られた値に補正をかけることが必要です。この操作は「各サンプルの遺伝子発現の強度分布はほとんど変わらない」という仮定が成り立つ時のみ意味を成すため、扱いには注意が必要です。


There are numerous proposals for normalizing unbalanced data between samples (:fa:`file-pdf-o` `This review paper summarizes 23 normalization methods developed for unbalanced transcriptome data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6988798/>`), but we will introduce only two of them.


1. Quantile
2. percen tile


****************
3. Summarization
****************

https://github.com/scipy/scipy/blob/v1.6.3/scipy/signal/signaltools.py#L3384-L3467

"""