# coding: utf-8
import numpy as np
import pandas as pd
from teilab.datasets import TeiLabDataSets

class TestData():
    def __init__(self):
        self.datasets = TeiLabDataSets()
        self.df_combined = self.create(self.datasets)

    def create(self, datasets):
        df_anno = datasets.read_data(no=0, usecols=datasets.ANNO_COLNAMES)
        reliable_index = set(df_anno.index)
        df_combined = df_anno.copy(deep=True)
        for no in range(2):
            df_data = datasets.read_data(no=no)
            reliable_index = reliable_index & set(datasets.reliable_filter(df=df_data))
            df_combined = pd.concat([
                df_combined, 
                df_data[[datasets.TARGET_COLNAME]].rename(columns={datasets.TARGET_COLNAME: datasets.samples.Condition[no]})
            ], axis=1)
        return df_combined.loc[reliable_index, :].reset_index(drop=True)

    @staticmethod
    def generate_normal_distributions(random_state=None, n_samples=4, n_features=1000):
        return np.random.RandomState(random_state).normal(loc=np.expand_dims(np.arange(n_samples), axis=1), size=(n_samples,n_features))
