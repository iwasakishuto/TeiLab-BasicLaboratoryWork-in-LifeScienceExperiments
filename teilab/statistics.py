#coding: utf-8
"""
**Statistical hypothesis testing** is required to determine if expression levels (``gProcessedSignal`` s) have changed between samples with siRNA and those without siRNA.

    A statistical hypothesis test is a method of statistical inference. An **alternative hypothesis** is proposed for the probability distribution of the data. The comparison of the two models is deemed statistically significant if, according to a threshold probability—the significance level ( :math:`\\alpha` ) — the data would be unlikely to occur if the **null hypothesis** were true. The **pre-chosen** level of significance is the maximal allowed "false positive rate". One wants to control the risk of incorrectly rejecting a true **null hypothesis**.

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
| right-tailed | ``"greater"``   |:math:`H_a:\sigma^2_1   >  \sigma^2_2`    | :math:`\mathbf{F}\geq F_{\\alpha}`                                               |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+
| left-tailed  | ``"less"``      |:math:`H_a:\sigma^2_1   <  \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\\alpha}`                                             |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+
| two-tailed   | ``"two-sided"`` |:math:`H_a:\sigma^2_1 \\neq \sigma^2_2`    | :math:`\mathbf{F}\leq F_{1−\\alpha∕2}` or :math:`\mathbf{F}\geq F_{\\alpha∕2}`    |
+--------------+-----------------+------------------------------------------+---------------------------------------------------------------------------------+

Follow the chart below to select the test.

.. graphviz:: _graphviz/graphviz_ChoosingStatisticalTest.dot
      :class: popup-img
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from typing import Optional,Any,Tuple
from numbers import Number
from matplotlib.axes import Axes
from nptyping import NDArray

from .utils.generic_utils import dict2str
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
    """F-Tests for Equality of Two Variances.

    If the two populations are normally distributed and if :math:`H_0:\sigma^2_1=\sigma^2_2` is true then under independent sampling :math:`F` approximately follows an F-distribution (:math:`f(x, df_1, df_2)`) with degrees of freedom :math:`df_1=n_1−1` and :math:`df_2=n_2−1`.

    The probability density function for `f` is:
    
    .. math::

        f(x, df_1, df_2) = \\frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{(df_2+df_1 x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}

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
        ...         ax.plot(x, f_dist.pdf(x, dfn=dfn, dfd=dfd), color="red", alpha=0.5, label="scipy")
        ...         ax.plot(x, f_pdf(x, dfn=dfn,dfd=dfd), color="blue", alpha=0.5, label="myfunc")
        ...         ax.set_title(f"f-distribution (dfn={dfn}, dfd={dfd})", fontsize=14)
        ...         ax.legend()
        >>> fig.show()

    +---------------------------------------------------+
    |                      Results                      |
    +===================================================+
    | .. image:: _images/statistics.f-distributions.jpg |
    |    :class: popup-img                              |
    +---------------------------------------------------+

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: Structure that holds test results.

    .. plot::
        :include-source:
        :class: popup-img

        >>> import numpy as np
        >>> from teilab.utils import subplots_create
        >>> from teilab.statistics import f_test
        >>> fig, axes = subplots_create(ncols=3, figsize=(18,4), style="matplotlib")
        >>> A = np.array([6.3, 8.1, 9.4, 10.4, 8.6, 10.5, 10.2, 10.5, 10.0, 8.8])
        >>> B = np.array([4.8, 2.1, 5.1, 2.0, 4.0, 1.0, 3.4, 2.7, 5.1, 1.4, 1.6])
        >>> for ax,alternative in zip(axes,["left","two","right"]):
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
        test_result.plot(x=np.linspace(0.001,max(x_max,f)+1,1000))
    return test_result

def paired_t_test(a:NDArray[Any, Number], b:NDArray[Any, Number], alpha:float=0.05, alternative:str="two-sided", plot:bool=False, ax:Optional[Axes]=None) -> TestResult:
    """T-Tests for Equality of Two averages of related samples.

    Args:
        a,b (NDArray[Any, Number])    : (Observed) Samples. The arrays must have the same shape.
        alpha (float)                 : The probability of making the wrong decision when the null hypothesis is true.
        alternative (str, optional)   : Defines the alternative hypothesis. Please choose from [ ``"two-sided"``, ``"less"``, ``"greater"`` ]. Defaults to ``"two-sided"``.
        plot (bool, optional)         : Whether to plot F-distribution or not. Defaults to ``False``.
        ax (Optional[Axes], optional) : An instance of ``Axes``. The distribution is drawn here when ``plot`` is ``True`` . Defaults to ``None``.

    Returns:
        TestResult: [description]

    .. seealso::
        https://en.wikipedia.org/wiki/T-test#Dependent_t-test_for_paired_samples
    """
    n = len(a)                   #: Sample size.
    df = n-1                     #: Degrees of Freedom.
    sd = np.sum(a-b)             #: Sum of Difference.
    ssd = np.sum(np.square(a-b)) #: Sum of Squared Difference.
    t = (sd/n) / np.sqrt( (ssd-(sd**2)/n)/((n-1)*n) ) #: T-score.
    t_dist = stats.t(df=df) #: T-distribution with given Degrees of Freedom.

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
        test_result.plot(x=np.linspace(-x_edge_abs, x_edge_abs, 1000))
    return test_result