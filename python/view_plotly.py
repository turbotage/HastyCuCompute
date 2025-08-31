import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html

# Generate spiral data
t = np.linspace(0, 10*np.pi, 500)

# Spiral 1 (standard helix)
x1 = np.cos(t)
y1 = np.sin(t)
z1 = t

# Spiral 2 (rotated along x-axis)
x2 = np.cos(t)
y2 = t / (2*np.pi)
z2 = np.sin(t)

# Spiral 3 (tilted)
x3 = t / (2*np.pi)
y3 = np.cos(t)
z3 = np.sin(t)

# Build 3D figure
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=x1, y=y1, z=z1,
    mode="lines",
    line=dict(color="red", width=4),
    name="Spiral 1"
))
fig.add_trace(go.Scatter3d(
    x=x2, y=y2, z=z2,
    mode="lines",
    line=dict(color="green", width=4),
    name="Spiral 2"
))
fig.add_trace(go.Scatter3d(
    x=x3, y=y3, z=z3,
    mode="lines",
    line=dict(color="blue", width=4),
    name="Spiral 3"
))

fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Spirals"
)

# Create Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Interactive 3D Spirals"),
    dcc.Graph(figure=fig, style={"height": "90vh"})  # Full height
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
