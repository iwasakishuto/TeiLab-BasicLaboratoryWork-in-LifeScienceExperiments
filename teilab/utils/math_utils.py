#coding: utf-8
import numpy as np

from typing import Any,Optional,Dict,List,Tuple
from nptyping import NDArray
from numbers import Number

def assign_rank(arr:NDArray[Any,Number], method:str="average") -> NDArray[Any,float]:
    """Assign rank to data, dealing with ties appropriately.

    Args:
        arr (NDArray[Any,Number]): The array of values to be ranked
        method (str, optional)   : The method used to assign ranks to tied elements. Defaults to ``"average"``.

    The following ``method``s are available.

    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | ``method``    | description                                                                                                                          |
    +===============+======================================================================================================================================+
    | ``"average"`` | The average of the ranks that would have been assigned to all the tied values is assigned to each value.                             |
    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | ``"min"``     | The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.                             |
    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | ``"max"``     | The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.                             |
    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | ``"dense"``   | Like ``"min"``, but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements. |
    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | ``"ordinal"`` | All values are given a distinct rank, corresponding to the order that the values occur in ``arr``                                    |
    +---------------+--------------------------------------------------------------------------------------------------------------------------------------+    

    Returns:
        NDArray[Any,float]: An array of size equal to the size of ``arr``, containing rank scores.

    Examples:
        >>> import numpy as np
        >>> from teilab.utils import assign_rank
        >>> arr = np.asarray([0,2,3,2])
        >>> assign_rank(arr, method="average")
        array([1. , 2.5, 4. , 2.5])
        >>> assign_rank(arr, method="min")
        array([1, 2, 4, 2])
        >>> assign_rank(arr, method="max")
        array([1, 3, 4, 3])
        >>> assign_rank(arr, method="dense")
        array([1, 2, 3, 2])
        >>> assign_rank(arr, method="ordinal")
        array([1, 2, 4, 3])

    .. seealso::
        - https://en.wikipedia.org/wiki/Ranking
        - `scipy.stats.rankdata(a, method='average') <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html>`_
    """
    sorter = np.argsort(arr, kind="mergesort" if method=="ordinal" else "quicksort") #: arr[sorter[i]] is i-th ranked values.
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp) #: inv[i] stores (original) arr[i]'s ranking (0-based index).
    if method == "ordinal":
        return inv+1 #: 1-based ranking.
    arr = arr[sorter] #: Sort the input array (arr)
    obs = np.r_[True, arr[1:] != arr[:-1]].astype(np.intp) #: obs[i] means arr[i]!=a[i-1]
    dense = obs.cumsum()[inv] #: dense[i] means (original) arr[i]'s ranking within unique values. (1-based index)
    if method == "dense":
        return dense
    count = np.r_[np.nonzero(obs)[0], len(obs)] # count[i] means the cumulative counts of unique values under i-th rank's unique value.
    if method == "max":
        return count[dense]
    if method == "min":
        return count[dense-1] + 1
    # average method
    return .5 * (count[dense] + count[dense-1] + 1)

def tiecorrect(ranks:NDArray[Any,Number]) -> float:
    """Tie correction factor for Mann-Whitney U and Kruskal-Wallis H tests.

    Args:
        ranks (NDArray[Any,Number]): A 1-D sequence of ranks. Typically this will be the array returned by :func:`assign_rank <teilab.utils.math_utils.assign_rank>` .

    Returns:
        float: Correction factor for ``U`` or ``H`` .

    Examples:
        >>> import numpy as np
        >>> from teilab.utils import tiecorrect, assign_rank
        >>> tiecorrect(np.asarray([0,2,3,2]))
        0.9
        >>> ranks = assign_rank(np.asarray([1,3,2,4,5,7,2,8,4]), method="average")
        >>> ranks
        array([1. , 4. , 2.5, 5.5, 7. , 8. , 2.5, 9. , 5.5])
        >>> tiecorrect(ranks)
        0.9833333333333333

    .. seealso::
        - :fa:`home` `Nonparametric Statistics for the Behavioral Sciences. <https://www.amazon.co.jp/-/en/Sidney-Siegel/dp/0070573484>`_
        - `scipy.stats.tiecorrect(rankvals) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tiecorrect.html>`_
    """
    arr  = np.sort(ranks)
    idx  = np.nonzero(np.r_[True, arr[1:] != arr[:-1], True])[0]
    cnt  = np.diff(idx).astype(np.float64)
    size = np.float64(arr.size)
    return 1.0 if size<2 else 1.0-(cnt**3-cnt).sum() / (size**3-size)
