import pandas as pd
from teilab.datasets import TeiLabDataSets
from teilab.plot.plotly import XYplot
datasets = TeiLabDataSets(verbose=False)
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
df_combined = df_combined.loc[reliable_index, :].reset_index(drop=True)
fig = XYplot(df=df_combined, x=datasets.samples.Condition[0], y=datasets.samples.Condition[1], hover_name="SystematicName", height=600, width=600)
fig.show()
