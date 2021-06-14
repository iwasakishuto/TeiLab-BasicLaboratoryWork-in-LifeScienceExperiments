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
        https://en.wikipedia.org/wiki/Statistical_hypothesis_testing

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
from .utils.generic_utils import dict2str
from .utils.math_utils import assign_rank, tiecorrect
from .utils.plot_utils import subplots_create
from .plot.matplotlib import update_layout

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
        https://en.wikipedia.org/wiki/F-test
    """
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

        - https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        - `scipy.stats.ttest_ind(A, B, equal_var=True) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_
    """
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

        - https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        - https://en.wikipedia.org/wiki/Welch%27s_t-test
        - `scipy.stats.ttest_ind(A, B, equal_var=False) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_
    """
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

        - https://en.wikipedia.org/wiki/T-test#Dependent_t-test_for_paired_samples
        - `scipy.stats.ttest_ind(A, B) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html>`_
    """
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

    .. admonition:: Statistic ( :math:`T` )
        
        .. container:: toggle, toggle-hidden

            Let :math:`A_{1},\ldots, A_{n}` be an i.i.d. sample from :math:`A`, and :math:`B_{1},\ldots ,B_{m}` be an i.i.d. sample from :math:`B`, and both samples independent of each other. The corresponding Mann-Whitney :math:`U` statistic is defined as:
    
            .. math::
                U=\sum_{i=1}^{n}\sum_{j=1}^{m}S(A_{i},B_{j})
            
            with

            .. math::
                S(A,B)={\begin{cases}1,&{\text{if }}B<A,\\{\tfrac {1}{2}},&{\text{if }}B=A,\\0,&{\text{if }}B>A.\end{cases}}


    Args:
        a,b (NDArray[Any, Number])      : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                   : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)     : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        use_continuity (bool, optional) : Whether a continuity correction ( ``1/2.`` ) should be taken into account. Default is ``True`` .
        plot (bool, optional)           : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional)   : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds Mann-Whitney's U-tesst results.

    .. seealso::

        - https://en.wikipedia.org/wiki/Mann-Whitney_U_test
        - :fa:`home` `On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other <https://www.jstor.org/stable/2236101>`_
        - `scipy.stats.mannwhitneyu(A, B) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html>`_
    """
    n_a = len(a)
    n_b = len(b)
    # Assign numeric ranks to all the observations (put the observations from both groups to one set), beginning with 1 for the samllest value.
    ranking = assign_rank(np.concatenate(a=(a,b)))
    rank_a = ranking[0:n_a] #: get the ``a``'s ranks
    u1 = n_a*n_b + (n_a*(n_a+1))/2.0 - np.sum(rank_a, axis=0) #: calculate ``U`` for ``a``
    u2 = n_a*n_b - u1 #: remainder is ``U`` for ``b``
    T = tiecorrect(ranking)
    if T == 0:
        raise ValueError('All numbers are identical in mannwhitneyu')
    sd = np.sqrt(T*n_a*n_b*(n_a+n_b+1) / 12.0)

    meanrank = n_a*n_b/2.0 + 0.5*use_continuity
    if alternative == 'two-sided':
        bigu = max(u1, u2)
    elif alternative == 'less':
        bigu = u1
    elif alternative == 'greater':
        bigu = u2

    z = (bigu - meanrank) / sd
    if alternative == 'two-sided':
        p = 2 * stats.norm.sf(abs(z))
    else:
        p = stats.norm.sf(z)

    u = u2
    return (u,p)

def wilcoxon_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    """NON-PARAMETRIC-test for Equality of averages of TWO PAIRED samples.

    .. admonition:: Statistic ( :math:`T` )
        
        .. container:: toggle, toggle-hidden
    
            .. math::
                XXX

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds wilcoxon-test results.

    .. seealso::

        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
        - `scipy.stats.wilcoxon(A, B) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html>`_
    """
    # warnings.warn(_pack_warning_args("message", __file__, sys._getframe().f_code.co_name), category=InsufficientUnderstandingWarning)

# def anova():

# def friedman_test():

# def kruskal_wallis_test():
