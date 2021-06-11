import plotly.graph_objects as go
from teilab.utils import subplots_create
from teilab.plot.plotly import update_layout
fig = subplots_create(nrows=1, ncols=2, style="plotly")
for c in range(1,3): fig.add_trace(go.Scatter(x=[1,2,3],y=[4,5,6]),row=1,col=c)
fig = update_layout(fig=fig, title="Sample", ylim=(4.5,5.5), col=2, height=400)
fig.show()
