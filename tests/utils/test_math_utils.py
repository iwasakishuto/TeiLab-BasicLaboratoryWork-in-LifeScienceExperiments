# coding: utf-8
import numpy as np


def test_assign_rank():
    from teilab.utils import assign_rank

    arr = np.asarray([0, 2, 3, 2])
    assert np.all(assign_rank(arr, method="average") == np.asarray([1.0, 2.5, 4.0, 2.5]))
    assert np.all(assign_rank(arr, method="min") == np.asarray([1, 2, 4, 2]))
    assert np.all(assign_rank(arr, method="max") == np.asarray([1, 3, 4, 3]))
    assert np.all(assign_rank(arr, method="dense") == np.asarray([1, 2, 3, 2]))
    assert np.all(assign_rank(arr, method="ordinal") == np.asarray([1, 2, 4, 3]))


def test_tiecorrect():
    from teilab.utils import assign_rank, tiecorrect

    tiecorrect(np.asarray([0, 2, 3, 2])) == 0.9
    ranks = assign_rank(np.asarray([1, 3, 2, 4, 5, 7, 2, 8, 4]), method="average")
    assert np.all(ranks == np.asarray([1.0, 4.0, 2.5, 5.5, 7.0, 8.0, 2.5, 9.0, 5.5]))
    assert tiecorrect(ranks) == 0.9833333333333333
