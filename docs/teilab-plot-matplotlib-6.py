from teilab.utils import subplots_create
from teilab.plot.matplotlib import update_layout
fig, axes = subplots_create(ncols=2, style="matplotlib", figsize=(8,4))
for ax in axes: ax.scatter(1,1,label="center")
_ = update_layout(ax=axes[1], xlim=(0,2), ylim=(0,2), legend=True)
fig.show()
