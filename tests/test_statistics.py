# coding: utf-8
import numpy as np
import pytest
from teilab.statistics import f_test, mann_whitney_u_test, paired_t_test, student_t_test, welch_t_test, wilcoxon_test
from teilab.utils import subplots_create

test_funcs = [f_test, student_t_test, welch_t_test, paired_t_test, mann_whitney_u_test, wilcoxon_test]


@pytest.mark.parametrize("test_func", test_funcs)
def _test_statistics(test_func: callable, alpha=0.05):
    fig, axes = subplots_create(ncols=3, figsize=(18, 4), style="matplotlib")
    rnd = np.random.RandomState(0)
    A, B = rnd.random_sample(size=(2, 30))
    for ax, alternative in zip(axes, ["less", "two-sided", "greater"]):
        test_func(A, B, alternative=alternative, plot=True, alpha=alpha, ax=ax)
    fig.show()
