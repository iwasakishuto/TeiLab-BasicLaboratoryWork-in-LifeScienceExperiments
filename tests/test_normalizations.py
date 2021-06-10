# coding: utf-8
import numpy as np
import pandas as pd

def test_percentile():
    from teilab.normalizations import percentile
    n_samples, n_features = (4, 1000)
    data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1),  size=(n_samples,n_features))
    data_percentiled = percentile(data=data, percent=75)
    assert data.shape == data_percentiled.shape

def test_quantile():
    from teilab.normalizations import quantile
    n_samples, n_features = (4, 1000)
    data = np.random.RandomState(0).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features), ) + 3.5
    data_quantiled = quantile(data=data)
    assert data.shape == data_quantiled.shape
    data_quantiled_sorted = np.sort(data_quantiled, axis=1)
    reference = data_quantiled_sorted[0,:].reshape(1,-1)
    assert np.all(reference == data_quantiled_sorted)

def test_median_polish():
    from teilab.normalizations import median_polish
    data = np.asarray([
        [16.1, 14.6, 19.6, 13.6, 13.6, 13.6],
        [ 9.0, 18.4,  6.7, 11.1,  6.7,  9.0],
        [22.4, 13.6, 22.4,  6.7,  9.0,  3.0],
    ], dtype=float).T
    n_samples, n_features = data.shape
    # This means that "All spots (features) are for the same gene."
    labels = np.zeros(shape=n_features, dtype=np.uint8)
    data_median_polished = median_polish(data, labels=labels)
    assert np.allclose(data_median_polished, np.asarray([
        [16.1 , 11.25, 13.3 ],
        [16.4 , 11.55, 13.6 ],
        [19.6 , 14.75, 16.8 ],
        [13.6 ,  8.75, 10.8 ],
        [11.8 ,  6.95,  9.  ],
        [13.6 ,  8.75, 10.8 ]
    ]))

def test_median_polish_group_wise():
    from teilab.normalizations import median_polish_group_wise
    df = pd.DataFrame(data=[
        ["vimentin", 16.1, 14.6, 19.6, 13.6, 13.6, 13.6],
        ["vimentin",  9.0, 18.4,  6.7, 11.1,  6.7,  9.0],
        ["vimentin", 22.4, 13.6, 22.4,  6.7,  9.0,  3.0],
    ], columns=["GeneName"]+[f"Samle.{i}" for i in range(6)])
    data_median_polished = df.groupby("GeneName").apply(func=median_polish_group_wise).values
    assert np.allclose(data_median_polished, np.asarray([
        [16.1 , 16.4 , 19.6 , 13.6 , 11.8 , 13.6 ],
        [11.25, 11.55, 14.75,  8.75,  6.95,  8.75],
        [13.3 , 13.6 , 16.8 , 10.8 ,  9.  , 10.8 ]
    ]))
