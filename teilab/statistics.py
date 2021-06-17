#coding: utf-8
r"""This submodule contains various functions and classes that are useful for statistical testing.

############
Instructions
############

******************************
Statistical hypothesis testing
******************************

**Statistical hypothesis testing** is required to determine if expression levels (``gProcessedSignal`` s) have changed between samples with siRNA and those without siRNA.

    A statistical hypothesis test is a method of statistical inference. An **alternative hypothesis** is proposed for the probability distribution of the data. The comparison of the two models is deemed statistically significant if, according to a threshold probability—the significance level ( :math:`\alpha` ) — the data would be unlikely to occur if the **null hypothesis** were true. The **pre-chosen** level of significance is the maximal allowed "false positive rate". One wants to control the risk of incorrectly rejecting a true **null hypothesis**.

    The process of distinguishing between the **null hypothesis** and the **alternative hypothesis** is aided by considering two conceptual types of errors.

    1. The first type of error occurs when the **null hypothesis** is wrongly rejected. (type 1 error)
    2. The second type of error occurs when the **null hypothesis** is wrongly not rejected. (type 2 error)

    Hypothesis tests based on statistical significance are another way of expressing confidence intervals (more precisely, confidence sets). In other words, every hypothesis test based on significance can be obtained via a confidence interval, and every confidence interval can be obtained via a hypothesis test based on significance. [#ref1]_

    .. [#ref1] :fa:`home` `Mathematical Statistics and Data Analysis (3rd ed.) <https://www.amazon.com/Mathematical-Statistics-Analysis-Available-Enhanced/dp/0534399428>`_

    .. seealso::
        :fa:`wikipedia-w` `Statistical_hypothesis_testing <https://en.wikipedia.org/wiki/Statistical_hypothesis_testing>`_

Various alternative hypotheses can be handled by changing the value of A

In this submodules, you can test each **null (alternative) hypothesis** by changing the value of ``alternative``

+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+
| Terminology  | ``alternative`` | Alternative Hypothesis                   | Rejection Region                                                                |
+==============+=================+==========================================+=================================================================================+
| right-tailed | ``"greater"``   |:math:`H_a:\sigma^2_1   >  \sigma^2_2`    | :math:`\mathbf{F}\geq F_{\alpha}`                                               |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+
| left-tailed  | ``"less"``      |:math:`H_a:\sigma^2_1   <  \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\alpha}`                                             |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+
| two-tailed   | ``"two-sided"`` |:math:`H_a:\sigma^2_1 \neq \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\alpha∕2}` or :math:`\mathbf{F}\geq F_{\alpha∕2}`    |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+

Follow the chart below to select the test.

.. graphviz:: _graphviz/graphviz_ChoosingStatisticalTest.dot
      :class: popup-img

Distributions
=============

- :ref:`gamma-distribution <target to gamma-distribution section>`
- :ref:`f-distribution <target to f-distribution section>`
- :ref:`t-distribution <target to t-distribution section>`

.. _target to gamma-distribution section:

gamma-distribution
------------------

The gamma function is defined as

    .. math::
       \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt
    
for :math:`\Re(z) > 0` and is extended to the rest of the complex plane by analytic continuation.

The gamma function is often referred to as the generalized factorial since :math:`\Gamma(n + 1) = n!` for natural numbers :math:`n`. More generally it satisfies the recurrence relation :math:`\Gamma(z + 1) = z \cdot \Gamma(z)` for complex :math:`z`, which, combined with the fact that :math:`\Gamma(1) = 1`, implies the above identity for :math:`z = n`.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import gamma
    >>> from scipy.integrate import quad
    >>> def gamma_pdf(x):
    ...     return quad(func=lambda x,z:np.power(x, z-1)*np.exp(-x), a=0, b=np.inf, args=(x))[0]
    >>> x = np.linspace(1e-2, 5, 100)
    >>> fig, ax = plt.subplots(figsize=(6, 4))
    >>> ax.plot(x, gamma(x), color="red", alpha=0.5, label="scipy")
    >>> ax.plot(x, np.vectorize(gamma_pdf)(x), color="blue", alpha=0.5, label="myfunc")
    >>> ax.set_title(fr"$\Gamma$-distribution", fontsize=14)
    >>> ax.legend()
    >>> ax.set_ylim(0, 10)
    >>> fig.savefig("statistics.distributions.gamma.jpg")

+-------------------------------------------------------+
|                      Results                          |
+=======================================================+
| .. image:: _images/statistics.distributions.gamma.jpg |
|    :class: popup-img                                  |
+-------------------------------------------------------+

.. _target to f-distribution section:

f-distribution
--------------

The probability density function for `f` is:

.. math::
    f(x, df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}

for :math:`x > 0`.

.. code-block:: python

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy import stats
        >>> from scipy.special import beta
        >>> def f_pdf(x,dfn,dfd):
        ...     return (dfd**(dfd/2) * dfn**(dfn/2) * x**(dfn/2-1)) / ((dfd+dfn*x)**((dfn+dfd)/2) * beta(dfn/2, dfd/2))
        >>> dfns = [2,5,12]; dfds = [2,9,12]
        >>> x = np.linspace(0.001, 5, 1000)
        >>> fig, axes = plt.subplots(nrows=len(dfns), ncols=len(dfds), sharex=True, sharey="row", figsize=(5*len(dfns), 5*len(dfds)))
        >>> for axes_row, dfn in zip(axes, dfns):
        ...     for ax, dfd in zip(axes_row, dfds):
        ...         ax.plot(x, stats.f.pdf(x, dfn=dfn, dfd=dfd), color="red", alpha=0.5, label="scipy")
        ...         ax.plot(x, f_pdf(x, dfn=dfn,dfd=dfd), color="blue", alpha=0.5, label="myfunc")
        ...         ax.set_title(f"f-distribution (dfn={dfn}, dfd={dfd})", fontsize=14)
        ...         ax.legend()
        >>> fig.show()

+---------------------------------------------------+
|                      Results                      |
+===================================================+
| .. image:: _images/statistics.distributions.f.jpg |
|    :class: popup-img                              |
+---------------------------------------------------+

.. _target to t-distribution section:

t-distribution
--------------

The probability density function for `t` is:

.. math::
    f(x, \nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu/2)}(1+x^2/\nu)^{-(\nu+1)/2}

where :math:`x` is a real number and the degrees of freedom parameter :math:`\nu` satisfies :math:`\nu > 0`.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> from scipy.special import gamma
    >>> def t_pdf(x,df):
    ...     return gamma((df+1)/2) / (np.sqrt(np.pi*df)*gamma(df/2)) * (1+x**2/df)**(-(df+1)/2)
    >>> x = np.linspace(-3, 3, 1000)
    >>> dfs = [1,3,7,90]
    >>> fig, axes = plt.subplots(nrows=1, ncols=len(dfs), sharey="row", figsize=(6*len(dfs), 4))
    >>> for ax,df in zip(axes,dfs):
    ...     ax.plot(x, stats.t.pdf(x, df=df), color="red", alpha=0.5, label="scipy")
    ...     ax.plot(x, t_pdf(x, df=df), color="blue", alpha=0.5, label="myfunc")
    ...     ax.set_title(f"t-distribution (df={df})", fontsize=14)
    ...     ax.legend()
    >>> fig.show()

+---------------------------------------------------+
|                      Results                      |
+===================================================+
| .. image:: _images/statistics.distributions.t.jpg |
|    :class: popup-img                              |
+---------------------------------------------------+

Notation
========

sample means
------------

Let

.. math::
    \overline{X}=\frac{1}{n_X}\sum_{i=1}^{n_X}X_{i}

be the sample means. 

sample variances
----------------

Let

.. math::
    S_X^2&={\frac{1}{n_X-1}}\sum_{i=1}^{n_X}\left(X_{i}-{\overline{X}}\right)^{2}\\&={\frac{1}{n_X-1}}\left(\sum_{i=1}^{n_X}X_i^2-n_X\overline{X}^2\right)

be the sample variances.

##############
Python Objects
##############
"""
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from typing import Optional,Any,Tuple
from numbers import Number
from matplotlib.axes import Axes
from nptyping import NDArray

from .utils._warnings import InsufficientUnderstandingWarning, TeiLabImprementationWarning, _pack_warning_args
from .utils import _wilcoxon_data
from .utils.generic_utils import dict2str, check_supported
from .utils.math_utils import assign_rank, tiecorrect
from .utils.plot_utils import subplots_create
from .plot.matplotlib import update_layout

SUPPORTED_ALTERNATIVES = ["two-sided", "less", "greater"]

class TestResult():
    """Structure that holds test results.

    Args:
        statistic (float)            : Test-specific statistical value.
        pvalue (float)               : The probability that an observed difference could have occurred just by random chance.
        alpha (float)                : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str)            : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]
        accepts (Tuple[float,float]) : Regions that accepts the null hypothesis.
        distribution (callable)      : Distribution used during the test.
        distname (str, optional)     : The distribution name. Defaults to ``""``.
        testname (str, optional)     : The test name. Defaults to ``""``.

    Attributes:
        does_H0_hold (bool) : Does the null hypothesis hold?
    """
    def __init__(self, statistic:float, pvalue:float, alpha:float, alternative:str, accepts:Tuple[float,float], distribution:callable, distname:str="", testname:str=""):
        self.statistic = statistic
        self.pvalue = pvalue
        self.alpha = alpha
        self.alternative = alternative
        self.accepts = accepts
        self.does_H0_hold = accepts[0]<statistic<accepts[1]
        self.distribution = distribution
        self.distname = distname
        self.testname = testname

    def __repr__(self):
        return f"{self.testname.capitalize()}_Result(statistic={self.statistic}, pvalue={self.pvalue})"

    def __iter__(self):
        for val in [self.statistic, self.pvalue]:
            yield val

    def plot(self, x:Optional[NDArray[Any, Number]]=None, ax:Optional[Axes]=None) -> Axes:
        """Plot the test result.

        Args:
            x (Optional[NDArray[Any, Number]], optional) : X-axis Values. Defaults to ``None``.
            ax (Optional[Axes], optional)                : An instance of ``Axes``. Defaults to ``None``.

        Returns:
            Axes: An instance of ``Axes`` with test result.
        """
        ax = ax or subplots_create(style="matplotlib")[1]
        if x is None:
            x = np.linspace(0,10,1000)
        y = self.distribution.pdf(x)
        # F-distribution
        ax.plot(x, y, color="blue", label=fr"{self.distname}-distribution ({dict2str(getattr(self.distribution, 'kwds', {}))})")
        # Visualize the Right and Left rejection areas
        ax.fill_between(x=x, y1=0, y2=y, where=(x<self.accepts[0]), color="red", label=f"Rejection Region (alpha={self.alpha})")
        ax.fill_between(x=x, y1=0, y2=y, where=(x>self.accepts[1]), color="red")
        # Indicates the position of calculated "F"
        stats_y = self.distribution.pdf(self.statistic)
        ax.scatter(x=self.statistic, y=stats_y, color="black", label=fr"$(x={self.distname.capitalize()}={self.statistic:.2f},y={stats_y:.2f})$")
        # Update Layout.
        ax = update_layout(ax=ax, title=f"{self.testname}-test (alternative: {self.alternative})\nNull-hypothesis {'holds' if self.does_H0_hold else 'rejects'}", xlabel="x", ylabel="y", legend=True)
        return ax

def f_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""F-test for Equality of TWO Variances.

    If the two populations are normally distributed and if :math:`H_0:\sigma^2_1=\sigma^2_2` is true then under independent sampling :math:`F` approximately follows an F-distribution (:math:`f(x, df_1, df_2)`) with degrees of freedom :math:`df_1=n_1−1` and :math:`df_2=n_2−1`.

    .. admonition:: Statistic ( :math:`F` )

        .. container:: toggle, toggle-hidden

            .. math::
                F={\frac  {S_{A}^{2}}{S_{B}^{2}}}

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds F-test results.

    Raises:
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import f_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
        >>> B = np.array([4.8, 2.1, 5.1,  2.0, 4.0,  1.0,  3.4,  2.7,  5.1, 1.4, 1.6])
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     f_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::
        :fa:`wikipedia-w` `F-test <https://en.wikipedia.org/wiki/F-test>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    a_unbiased_var = np.var(a, ddof=1) #: ``a``'s unbiased variable. 
    b_unbiased_var = np.var(b, ddof=1) #: ``b``'s unbiased variable. 
    a_degrees_of_freedom = len(a)-1    #: ``a``'s Degrees of Freedom.
    b_degrees_of_freedom = len(b)-1    #: ``b``'s Degrees of Freedom.
    f = a_unbiased_var / b_unbiased_var #: F-score.
    f_dist = stats.f(dfn=a_degrees_of_freedom, dfd=b_degrees_of_freedom) # F-distribution with given Degrees of Freedoms.

    # ppf : Percent point function 
    # cdf : Cumulative distribution function
    # sf  : Survival function (1 - cdf )
    if alternative == "less":
        x_l = f_dist.ppf(alpha)
        x_r = f_dist.ppf(1)
        p_val = f_dist.cdf(f)
    elif alternative == "greater":
        x_l = f_dist.ppf(0)
        x_r = f_dist.ppf(1-alpha)
        p_val = f_dist.sf(f)
    else: # Two-side
        x_l = f_dist.ppf(alpha/2)
        x_r = f_dist.ppf(1-alpha/2)
        p_val = 2*min(f_dist.cdf(f), f_dist.sf(f))

    test_result = TestResult(
        statistic=f, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
        distribution=f_dist, distname="F", testname="f"
    )
    if plot:
        x_max = 3
        while True:
            if f_dist.pdf(x_max)<1e-3: break
            x_max+=1
        test_result.plot(x=np.linspace(0.001,max(x_max,f)+1,1000), ax=ax)
    return test_result

def student_t_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""T-test for Equality of averages of TWO INDEPENDENT samples. (SIMILAR VARIANCES)

    .. admonition:: Statistic ( :math:`T` )
        
        .. container:: toggle, toggle-hidden
    
            .. math::
                T={\frac{\overline{A}-{\overline{B}}}{s_p\cdot {\sqrt {\frac{1}{n_A}+\frac{1}{n_B}}}}}

            where

            .. math::
                s_p=\sqrt{\frac {\left(n_A-1\right)s_A^2+\left(n_B-1\right)s_B^2}{n_A+n_B-2}}

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds student's T-test results.

    Raises:
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import student_t_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
        >>> B = np.array([4.8, 2.1, 5.1,  2.0, 4.0,  1.0,  3.4,  2.7,  5.1, 1.4, 1.6])
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     student_t_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::

        - :fa:`wikipedia-w` `T-test <https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test>`_
        - `scipy.stats.ttest_ind(A, B, equal_var=True) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    n_a = len(a) #: Sample size of ``a``.
    n_b = len(b) #: Sample size of ``b``.
    df = (n_a-1)+(n_b-1) #: Degrees of Freedom.
    sum_a = np.sum(a) # Sum of ``a``.
    sum_b = np.sum(b) # Sum of ``b``.
    t = (sum_a/n_a - sum_b/n_b) / np.sqrt( ((np.sum(np.square(a))-np.square(sum_a)/n_a) + (np.sum(np.square(b))-np.square(sum_b)/n_b))/df * (1/n_a+1/n_b) ) #: T-score.
    t_dist = stats.t(df=df) #: T-distribution with given Degrees of Freedom.

    # ppf : Percent point function 
    # cdf : Cumulative distribution function
    # sf  : Survival function (1 - cdf )
    if alternative == "less":
        x_l = t_dist.ppf(alpha)
        x_r = t_dist.ppf(1)
        p_val = t_dist.cdf(t)
    elif alternative == "greater":
        x_l = t_dist.ppf(0)
        x_r = t_dist.ppf(1-alpha)
        p_val = t_dist.sf(t)
    else: # Two-side
        x_l = t_dist.ppf(alpha/2)
        x_r = t_dist.ppf(1-alpha/2)
        p_val = 2*t_dist.sf(abs(t))
    
    test_result = TestResult(
        statistic=t, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
        distribution=t_dist, distname="T", testname="student's-t"
    )
    if plot:
        x_edge_abs = max(t_dist.ppf(1e-4), abs(t)+1)
        test_result.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000), ax=ax)
    return test_result

def welch_t_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""T-test for Equality of averages of **TWO INDEPENDENT** samples. (**DIFFERENT VARIANCES**)

    .. admonition:: Statistic ( :math:`T` )
        
        .. container:: toggle, toggle-hidden
    
            .. math::
                T=\frac{\Delta\overline {X}}{s_{\Delta\overline{X}}}=\frac{\overline{X}_1-\overline{X}_2}{\sqrt{{s^2_{\overline{X}_1}}+s^2_{\overline{X}_{2}}}}

            where

            .. math::
                s_{{\overline{X}}_{i}}={s_i\over{\sqrt{N_i}}}

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds welch's T-test results.

    Raises:
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import welch_t_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
        >>> B = np.array([4.8, 2.1, 5.1,  2.0, 4.0,  1.0,  3.4,  2.7,  5.1, 1.4, 1.6])
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     welch_t_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::

        - :fa:`wikipedia-w` `T-test <https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test>`_
        - :fa:`wikipedia-w` `Welch's_t-test <https://en.wikipedia.org/wiki/Welch%27s_t-test>`_
        - `scipy.stats.ttest_ind(A, B, equal_var=False) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    vna = np.var(a, ddof=1)/len(a) # ``a``'s unbiased variable. devided by len(a)
    vnb = np.var(b, ddof=1)/len(b) # ``b``'s unbiased variable. devided by len(b)
    df = (vna+vnb)**2 / (vna**2/(len(a)-1) + vnb**2/(len(b)-1)) #: Degrees of Freedom.
    t = (np.mean(a)-np.mean(b))/np.sqrt(vna+vnb) #: T-score.
    t_dist = stats.t(df=df) #: T-distribution with given Degrees of Freedom.
    
    # ppf : Percent point function 
    # cdf : Cumulative distribution function
    # sf  : Survival function (1 - cdf )
    if alternative == "less":
        x_l = t_dist.ppf(alpha)
        x_r = t_dist.ppf(1)
        p_val = t_dist.cdf(t)
    elif alternative == "greater":
        x_l = t_dist.ppf(0)
        x_r = t_dist.ppf(1-alpha)
        p_val = t_dist.sf(t)
    else: # Two-side
        x_l = t_dist.ppf(alpha/2)
        x_r = t_dist.ppf(1-alpha/2)
        p_val = 2*t_dist.sf(abs(t))
    
    test_result = TestResult(
        statistic=t, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
        distribution=t_dist, distname="T", testname="welch-t"
    )

    if plot:
        x_edge_abs = max(t_dist.ppf(1e-4), abs(t)+1)
        test_result.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000), ax=ax)
    return test_result

def paired_t_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""T-test for Equality of averages of TWO RELATED samples.

    .. admonition:: Statistic ( :math:`T` )
        
        .. container:: toggle, toggle-hidden

            .. math::
                T=\frac{\overline{D}}{S_D/\sqrt{N}}

            where :math:`D` is the sample differences ( :math:`D=A-B` ), :math:`N` is the sample length ( :math:`N=\mathrm{len}(A)=\mathrm{len}(B)`)

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds paired T-test results.

    Raises:
        TypeError: When the arrays ``a`` and ``b`` have the different shapes.
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import paired_t_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
        >>> B = np.array([1.9,  0.8,  1.1,  0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     paired_t_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::

        - :fa:`wikipedia-w` `T-test <https://en.wikipedia.org/wiki/T-test#Dependent_t-test_for_paired_samples>`_
        - `scipy.stats.ttest_ind(A, B) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    n = len(a) #: Sample size.
    if n!=len(b):
        raise TypeError(f"The arrays (a,b) must have the same shape, but got {n}!={len(b)}")
    t = np.mean(a-b)/np.sqrt(np.var(a-b, ddof=1)/n) #: T-score.
    t_dist = stats.t(df=n-1) #: T-distribution with given Degrees of Freedom.

    # ppf : Percent point function 
    # cdf : Cumulative distribution function
    # sf  : Survival function (1 - cdf )
    if alternative == "less":
        x_l = t_dist.ppf(alpha)
        x_r = t_dist.ppf(1)
        p_val = t_dist.cdf(t)
    elif alternative == "greater":
        x_l = t_dist.ppf(0)
        x_r = t_dist.ppf(1-alpha)
        p_val = t_dist.sf(t)
    else: # Two-side
        x_l = t_dist.ppf(alpha/2)
        x_r = t_dist.ppf(1-alpha/2)
        p_val = 2*t_dist.sf(abs(t))
    
    test_result = TestResult(
        statistic=t, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
        distribution=t_dist, distname="T", testname="paired-t"
    )

    if plot:
        x_edge_abs = max(t_dist.ppf(1e-4), abs(t)+1)
        test_result.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000), ax=ax)
    return test_result

def mann_whitney_u_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", use_continuity:bool=True, plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""NON-PARAMETRIC-test for Equality of averages of TWO INDEPENDENT samples.

    Mann–Whitney U test is used as significance


    .. admonition:: Statistic ( :math:`U` )
        
        .. container:: toggle, toggle-hidden

            Let :math:`A_{1},\ldots, A_{n}` be an i.i.d. sample from :math:`A`, and :math:`B_{1},\ldots ,B_{m}` be an i.i.d. sample from :math:`B`, and both samples independent of each other. The corresponding Mann-Whitney :math:`U` statistic is defined as:
    
            .. math::
                U=\sum_{i=1}^{n_A}\sum_{j=1}^{n_B}S(A_{i},B_{j})
            
            with

            .. math::
                S(A,B)={\begin{cases}1,&{\text{if }}B<A,\\{\tfrac {1}{2}},&{\text{if }}B=A,\\0,&{\text{if }}B>A.\end{cases}}

            In direct method, for each observation in sample ``a``, the sum of the number of observations in sample ``b`` for which a smaller value was obtained is the :math:`U_A` .

            In other method, :math:`U_A` is given by:

            .. math::
                U_A=R_A-\frac{n_A(n_A+1)}{2}

            where :math:`n_A` is the sample size for sample ``a``, and :math:`R_A` is the sum of the ranks in sample ``a`` because :math:`\frac{n_A(n_A+1)}{2}` is the sum of all the ranks of ``a`` within sample ``a``.

            Knowing that :math:`R_A+R_B=N(N+1)/2` and :math:`N=n_A+n_B`, we find that the sum is

            .. math::
                U_A + U_B 
                &= \left(R_A - \frac{n_A(n_A+1)}{2}\right) + \left(R_B - \frac{n_B(n_B+1)}{2}\right) \\ 
                &= \underbrace{\left(R_A + R_B\right)}_{(N)(N+1)/2} - \frac{1}{2}\left(n_A(n_A+1) + n_B(n_B+1)\right)\\
                &= n_An_B.

            For large ( :math:`>20` ) samples, :math:`U` is approximately normally distributed. In this case, the standardized value

            .. math::
                z={\frac {U-m_{U}}{\sigma _{U}}}

            where :math:`m_U` and :math:`\sigma_U` are the mean and standard deviation of :math:`U`, is approximately a standard normal deviate whose significance can be checked in tables of the normal distribution. :math:`m_U` and :math:`\sigma_U` are given by

            .. math::
                m_U={\frac {n_An_B}{2}},\ \text{and} \\
                \sigma _{U}={\sqrt {\frac{n_An_B(n_A+n_B+1)}{12}}}.

            The formula for the standard deviation is more complicated in the presence of tied ranks. If there are ties in ranks, :math:`\sigma` should be corrected as follows:

            .. math::
                \sigma _{\text{corr}}={\sqrt {\frac{n_An_B}{12}\left((N+1)-\sum_{i=1}^k\frac{t_i^3-t_i}{N(N-1)}\right)}}

            where :math:`N=n_A+n_B, t_i` is the number of subjects sharing rank :math:`i`, and :math:`k` is the number of (distinct) ranks.

    .. warning::
        How to calculate the statistic value is different from `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html>`_.

    Args:
        a,b (NDArray[Any, Number])      : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                   : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)     : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        use_continuity (bool, optional) : Whether a continuity correction ( ``1/2.`` ) should be taken into account. Default is ``True`` .
        plot (bool, optional)           : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional)   : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds Mann-Whitney's U-tesst results.

    Raises:
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].
        ValueError: When all numbers are identical in mannwhitneyu.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import mann_whitney_u_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([1.83, 1.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 2.01, 3.11])
        >>> B = np.array([0.88, 0.65, 0.60, 1.05, 1.06, 1.29, 1.06, 2.14, 1.29])
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     mann_whitney_u_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::

        - :fa:`wikipedia-w` `Mann-Whitney_U_test <https://en.wikipedia.org/wiki/Mann-Whitney_U_test>`_
        - :fa:`home` `On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other <https://www.jstor.org/stable/2236101>`_
        - `scipy.stats.mannwhitneyu(A, B) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    n_a = len(a)
    n_b = len(b)
    # Assign numeric ranks to all the observations (put the observations from both groups to one set), beginning with 1 for the samllest value.
    ranking = assign_rank(np.concatenate((a,b)))
    # Calculate Ua (for ``a``) and Ub (for ``b``)
    Ua = np.sum(ranking[0:n_a]) - (n_a*(n_a+1))/2.0
    Ub = n_a*n_b - Ua #: ∵) Ua+Ub = n_a*n_b, so remainder is ``Ub``
    T = tiecorrect(ranking)
    if T == 0:
        raise ValueError('All numbers are identical in mannwhitneyu')
    mean_rank = n_a*n_b/2.0 + 0.5*use_continuity    #: mean of ``U``
    sd_rank = np.sqrt(T*n_a*n_b*(n_a+n_b+1) / 12.0) #: corrected standard deviation of ``U``
    norm_dist = stats.norm(loc=0, scale=1)

    if alternative == "two-sided":
        u = min(Ua,Ub)
        z = (u-mean_rank) / sd_rank
        x_l = norm_dist.ppf(alpha/2)
        x_r = norm_dist.ppf(1-alpha/2)
        p_val = 2*norm_dist.sf(abs(z))
    else:
        u = Ua
        z = (u-mean_rank) / sd_rank
        if alternative == "less":
            x_l = norm_dist.ppf(alpha)
            x_r = norm_dist.ppf(1)
            p_val = norm_dist.cdf(z)
        else: # alternative == "greater":
            x_l = norm_dist.ppf(0)
            x_r = norm_dist.ppf(1-alpha)
            p_val = norm_dist.sf(z)
    
    test_result = TestResult(
        statistic=u, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
        distribution=None, distname="(approx.) Normal Distribution", testname="Mann-Whitney's-u"
    )
    if plot:
        test_result_for_plot = TestResult(
            statistic=z, pvalue=p_val, 
            alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
            distribution=norm_dist, distname="Normal", testname="Mann-Whitney's-u"
        )
        x_edge_abs = max(norm_dist.ppf(1e-4), abs(z)+1)
        test_result_for_plot.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000), ax=ax)
    return test_result

def wilcoxon_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", mode:str="auto", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    r"""NON-PARAMETRIC-test for Equality of representative values (medians) of TWO PAIRED samples.

    This function is about the Wilcoxon **signed-rank** test which tests whether the distribution of the differences ``a - b`` is symmetric about zero. It is a non-parametric version of the :func:`paired_t_test <teilab.statistics.paired_t_test>`.

    NOTE:
        Wilcoxon **signed-rank** test is different from Wilcoxon **rank sum** test which is equivalent to the :func:`student_t_test <teilab.statistics.student_t_test>` or :func:`welch_t_test <teilab.statistics.welch_t_test>` in the parametric test.
    
    .. admonition:: Statistic ( :math:`W` )
        
        .. container:: toggle, toggle-hidden
    
            Let :math:`N` be the sample size, i.e., the number of pairs (:math:`A=a_i,\ldots,a_N, B=b_i,\ldots,b_N`).

            1. Exclude pairs with :math:`|a_i-b_i|=0`. Let :math:`N_r` be the reduced sample size.
            2. Order the remaining :math:`N_r` pairs acoording to the absolute difference (:math:`|a_i-b_i|`).
            3. Rank the pairs. Ties receive a rank equal to the average og the ranks they span (:func:`assign_rank(a-b, method="average") <teilab.utils.math_utils.assign_rank>`). Let :math:`R_i` denote the rank.
            4. Calculate the test statistic :math:`W` (the sum of the signed ranks)

            .. math::
                W_{+}&=\sum_{i=1}^{N_r}\left[\operatorname{sgn}(a_i-b_i)\cdot R_{i}\right],\\
                W_{-}&=\sum_{i=1}^{N_r}\left[-\operatorname{sgn}(a_i-b_i)\cdot R_{i}\right].

            5. From the above, use :math:`W` that matches the alternative hypothesis.

            .. math::
                W=\begin{cases}
                \mathrm{min}\left(W_{+},W_{-}\right),&\text{if alternative hypothesis is "two-tailed"},\\
                W_{+},&\text{otherwise.}
                \end{cases}
            
            6. As :math:`N_r` increases, the sampling distribution of :math:`W` converges to a normal distribution. Thus, 
                
                1. For :math:`N_r\geq25`, a z-score can be calculated as 
                
                .. math::
                    Z = \frac{\left|W-\frac{N_r(N_r+1)}{4}\right|}{\sqrt{\frac{N_r(N_r+1)(2N_r+1)}{24}}}
                
                2. For :math:`N_r<25` the exact distribution needs to be used.

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        mode (str, optional)          : Method to calculate the p-value. Please choose from [ ``"auto"``, ``"exact"``, ``"approx"`` ]. Default is "auto".
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds wilcoxon-test results.

    Raises:
        TypeError: When the arrays ``a`` and ``b`` have the different shapes.
        KeyError: When ``alternative`` is not selected from [ ``"two-sided"``, ``"less"``, ``"greater"`` ].
        ValueError: When all pairs are identical, or try to calculate ``"exact"`` p-value with the sample size larger than ``25``.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import wilcoxon_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> rnd = np.random.RandomState(0)
        >>> A,B = rnd.random_sample(size=(2,30))
        >>> for ax,alternative in zip(axes,["less","two-sided","greater"]):
        ...     wilcoxon_test(A, B, alternative=alternative, plot=True, alpha=.1, ax=ax)
        >>> fig.show()

    .. seealso::

        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
        - `scipy.stats.wilcoxon(A, B, correction=False, alternative="two-sided") <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html>`_
        - :fa:`wikipedia-w` `Wilcoxon_signed-rank_test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_
    """
    check_supported(lst=SUPPORTED_ALTERNATIVES, alternative=alternative)
    check_supported(lst=["auto","exact","approx"], mode=mode)
    n = len(a)
    if n!=len(b):
        raise TypeError(f"The arrays (a,b) must have the same shape, but got {n}!={len(b)}")
    # Determine how to calculate the p-value from the difference and the samle size.
    diff = a-b
    if mode=="auto":
        if n <= 25:
            mode="exact"
        else:
            mode="approx"
    if np.any(diff==0) and mode=="exact":
        mode = "approx"
        warnings.warn(_pack_warning_args("Exact p-value calculation does not work if there are ties. Switching to normal approximation.", __file__, sys._getframe().f_code.co_name), category=InsufficientUnderstandingWarning)
    # Leave only pairs with differences.
    diff = diff[np.nonzero(diff)[0]]
    count = len(diff)
    if count==0:
        raise ValueError("All pairs are identical.")
    if count<10 and mode=="approx":
        warnings.warn(_pack_warning_args("Sample size too samll for normal approximation", __file__, sys._getframe().f_code.co_name), category=InsufficientUnderstandingWarning)
    # Assign Ranking and calculate T-score.
    ranking = assign_rank(np.abs(diff))
    w_plus  = np.sum((diff>0)*ranking)
    w_minus = np.sum((diff<0)*ranking)
    if alternative == "two-sided":
        W = min(w_plus, w_minus)
    else:
        W = w_plus
    # Calculate p-value.
    if mode=="approx":
        norm_dist = stats.norm(loc=0, scale=1)
        z = (W-count*(count+1.)*0.25) / np.sqrt(count*(count+1.)*(2.*count+1.)/24)
        if alternative == "two-sided":
            x_l = norm_dist.ppf(alpha/2)
            x_r = norm_dist.ppf(1-alpha/2)
            p_val = 2*norm_dist.sf(abs(z))
        elif alternative == "less":
            x_l = norm_dist.ppf(alpha)
            x_r = norm_dist.ppf(1)
            p_val = norm_dist.cdf(z)
        else:
            x_l = norm_dist.ppf(0)
            x_r = norm_dist.ppf(1-alpha)
            p_val = norm_dist.sf(z)
        if plot:
            test_result_for_plot = TestResult(
                statistic=z, pvalue=p_val, 
                alpha=alpha, alternative=alternative, accepts=(x_l,x_r), 
                distribution=norm_dist, distname="(approx.) Normal", testname="Wilcoxon signed-rank"
            )
            x_edge_abs = max(norm_dist.ppf(1e-4), abs(z)+1)
            test_result_for_plot.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000), ax=ax)
    else:
        cnt = _wilcoxon_data.COUNTS.get(count)
        if cnt is None:
            raise ValueError(f"The exact distribution of the Wilcoxon test static is not implemented for n={count}")
        cnt = np.asarray(cnt, dtype=int)
        # note: w_plus is int (ties not allowed), need int for slices below
        w_plus = int(w_plus)
        if alternative == "two-sided":
            if w_plus == (len(cnt) - 1) // 2:
                # w_plus is the center of the distribution.
                prob = 1.0
            else:
                p_less = np.sum(cnt[:w_plus + 1]) / 2**count
                p_greater = np.sum(cnt[w_plus:]) / 2**count
                p_val = 2*min(p_greater, p_less)
        elif alternative == "greater":
            p_val = np.sum(cnt[w_plus:]) / 2**count
        else:
            p_val = np.sum(cnt[:w_plus + 1]) / 2**count
        # if plot:
        #     warnings.warn()
    test_result = TestResult(
        statistic=W, pvalue=p_val, 
        alpha=alpha, alternative=alternative, accepts=(0, 1), 
        distribution=None, distname="(approx.) Normal", testname="Wilcoxon signed-rank"
    )
    return test_result

def anova() -> TestResult:
    raise NotImplementedError("I'm sorry... Not Impremented.")

def friedman_test() -> TestResult:
    raise NotImplementedError("I'm sorry... Not Impremented.")

def kruskal_wallis_test() -> TestResult:
    raise NotImplementedError("I'm sorry... Not Impremented.")
