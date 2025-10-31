import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np


class MedicalSlicer:
    def __init__(self, volume):
        """
        Initialize the medical slicer with a volume.

        Args:
            volume: numpy array of shape (X, Y, Z), (X, Y, Z, T), or (X, Y, Z, T, C)
        """
        self.volume = volume
        self.ndim = volume.ndim
        self.shape = volume.shape

        # Initialize slice indices
        self.sagittal_idx = self.shape[0] // 2 if self.ndim >= 3 else 0
        self.coronal_idx = self.shape[1] // 2 if self.ndim >= 3 else 0
        self.axial_idx = self.shape[2] // 2 if self.ndim >= 3 else 0

    def get_slice(self, view, i_slice, j_slice, k_slice, t_idx=0, c_idx=0):
        """Get a 2D slice from the volume."""
        if self.ndim == 2:
            return self.volume
        elif self.ndim == 3:
            if view == 'sagittal':
                return self.volume[i_slice, :, :]
            elif view == 'coronal':
                return self.volume[:, j_slice, :]
            elif view == 'axial':
                return self.volume[:, :, k_slice]
        elif self.ndim == 4:
            if view == 'sagittal':
                return self.volume[i_slice, :, :, t_idx]
            elif view == 'coronal':
                return self.volume[:, j_slice, :, t_idx]
            elif view == 'axial':
                return self.volume[:, :, k_slice, t_idx]
        elif self.ndim == 5:
            if view == 'sagittal':
                return self.volume[i_slice, :, :, t_idx, c_idx]
            elif view == 'coronal':
                return self.volume[:, j_slice, :, t_idx, c_idx]
            elif view == 'axial':
                return self.volume[:, :, k_slice, t_idx, c_idx]

        return np.zeros((10, 10))


def create_dash_app(volume):
    """Create and configure the Dash application."""
    slicer = MedicalSlicer(volume)
    app = dash.Dash(__name__)

    # Build controls - only for dimensions 4 and 5
    controls = []
    
    # Position labels
    if slicer.ndim >= 3:
        controls.append(html.Div([
            html.Label(f'Sagittal (I): {slicer.sagittal_idx}', id='label-i', style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Label(f'Coronal (J): {slicer.coronal_idx}', id='label-j', style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Label(f'Axial (K): {slicer.axial_idx}', id='label-k', style={'display': 'inline-block'}),
        ], style={'margin': '10px', 'textAlign': 'center', 'fontSize': '16px', 'fontWeight': 'bold'}))

    # Only add sliders for 4th and 5th dimensions
    if slicer.ndim >= 4:
        controls.append(html.Div([
            html.Label(f'Time (T): 0 / {slicer.shape[3]-1}', id='label-t'),
            dcc.Slider(
                id='t-slider',
                min=0,
                max=slicer.shape[3]-1,
                value=0,
                marks={i: str(i) for i in range(0, slicer.shape[3], max(1, slicer.shape[3]//5))},
                step=1,
            )
        ], style={'margin': '10px'}))

    if slicer.ndim >= 5:
        controls.append(html.Div([
            html.Label(f'Channel (C): 0 / {slicer.shape[4]-1}', id='label-c'),
            dcc.Slider(
                id='c-slider',
                min=0,
                max=slicer.shape[4]-1,
                value=0,
                marks={i: str(i) for i in range(0, slicer.shape[4], max(1, slicer.shape[4]//5))},
                step=1,
            )
        ], style={'margin': '10px'}))

    # Hidden stores for slice indices
    stores = [
        dcc.Store(id='store-i', data=slicer.sagittal_idx),
        dcc.Store(id='store-j', data=slicer.coronal_idx),
        dcc.Store(id='store-k', data=slicer.axial_idx),
    ]

    # Create views
    if slicer.ndim == 2:
        views = html.Div([
            dcc.Graph(id='view-2d', style={'height': '600px'}, config={'displayModeBar': False})
        ])
    else:
        views = html.Div([
            dcc.Graph(id='sagittal-view', style={'height': '500px', 'width': '33%', 'display': 'inline-block'}, config={'displayModeBar': False}),
            dcc.Graph(id='coronal-view', style={'height': '500px', 'width': '33%', 'display': 'inline-block'}, config={'displayModeBar': False}),
            dcc.Graph(id='axial-view', style={'height': '500px', 'width': '33%', 'display': 'inline-block'}, config={'displayModeBar': False}),
        ])

    app.layout = html.Div([
        html.H1('Medical Image Slicer', style={'textAlign': 'center'}),
        html.Div('Click on a view to navigate through the volume', style={'textAlign': 'center', 'color': '#666', 'marginBottom': '10px'}),
        html.Div(controls, style={'padding': '20px'}),
        html.Div(stores),
        views,
    ])

    def make_figure(data, title):
        """Create a clean 2D image figure using go.Heatmap."""
        fig = go.Figure(go.Heatmap(
            z=data,
            colorscale='Gray',
            showscale=True,
            hoverinfo='skip',
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                scaleanchor='y',
                scaleratio=1,
                fixedrange=True,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                fixedrange=True,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    # Callbacks
    if slicer.ndim == 2:
        @app.callback(
            Output('view-2d', 'figure'),
            Input('view-2d', 'id')
        )
        def update_2d(_):
            data = slicer.get_slice('axial', 0, 0, 0)
            return make_figure(data, '2D Image')
    else:
        # Update slice indices from clicks
        @app.callback(
            [Output('store-i', 'data'),
             Output('store-j', 'data'),
             Output('store-k', 'data')],
            [Input('sagittal-view', 'clickData'),
             Input('coronal-view', 'clickData'),
             Input('axial-view', 'clickData')],
            [State('store-i', 'data'),
             State('store-j', 'data'),
             State('store-k', 'data')]
        )
        def update_indices(sag_click, cor_click, ax_click, i_val, j_val, k_val):
            ctx = dash.callback_context
            if not ctx.triggered:
                return i_val, j_val, k_val
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'sagittal-view' and sag_click:
                # Sagittal shows J-K plane, clicking sets J and K
                j_val = int(sag_click['points'][0]['x'])
                k_val = int(sag_click['points'][0]['y'])
            elif trigger_id == 'coronal-view' and cor_click:
                # Coronal shows I-K plane, clicking sets I and K
                i_val = int(cor_click['points'][0]['x'])
                k_val = int(cor_click['points'][0]['y'])
            elif trigger_id == 'axial-view' and ax_click:
                # Axial shows I-J plane, clicking sets I and J
                i_val = int(ax_click['points'][0]['x'])
                j_val = int(ax_click['points'][0]['y'])
            
            return i_val, j_val, k_val
        
        # Update views
        @app.callback(
            [Output('sagittal-view', 'figure'),
             Output('coronal-view', 'figure'),
             Output('axial-view', 'figure'),
             Output('label-i', 'children'),
             Output('label-j', 'children'),
             Output('label-k', 'children')] +
            ([Output('label-t', 'children')] if slicer.ndim >= 4 else []) +
            ([Output('label-c', 'children')] if slicer.ndim >= 5 else []),
            [Input('store-i', 'data'),
             Input('store-j', 'data'),
             Input('store-k', 'data'),
             Input('t-slider', 'value') if slicer.ndim >= 4 else Input('store-i', 'data'),
             Input('c-slider', 'value') if slicer.ndim >= 5 else Input('store-i', 'data')],
        )
        def update_views(i_val, j_val, k_val, t_val, c_val):
            t_idx = t_val if slicer.ndim >= 4 else 0
            c_idx = c_val if slicer.ndim >= 5 else 0

            # Get slices
            sag_data = slicer.get_slice('sagittal', i_val, j_val, k_val, t_idx, c_idx)
            cor_data = slicer.get_slice('coronal', i_val, j_val, k_val, t_idx, c_idx)
            ax_data = slicer.get_slice('axial', i_val, j_val, k_val, t_idx, c_idx)

            # Create figures
            sag_fig = make_figure(sag_data, f'Sagittal (I={i_val})')
            cor_fig = make_figure(cor_data, f'Coronal (J={j_val})')
            ax_fig = make_figure(ax_data, f'Axial (K={k_val})')

            # Update labels
            label_i = f'Sagittal (I): {i_val}'
            label_j = f'Coronal (J): {j_val}'
            label_k = f'Axial (K): {k_val}'

            outputs = [sag_fig, cor_fig, ax_fig, label_i, label_j, label_k]

            if slicer.ndim >= 4:
                outputs.append(f'Time (T): {t_idx} / {slicer.shape[3]-1}')
            if slicer.ndim >= 5:
                outputs.append(f'Channel (C): {c_idx} / {slicer.shape[4]-1}')

            return outputs

    return app


# Example usage
if __name__ == '__main__':
    # Create sample 3D volume
    volume = np.random.rand(100, 100, 100)

    # For 4D: volume = np.random.rand(100, 100, 100, 20)
    # For 5D: volume = np.random.rand(100, 100, 100, 20, 3)

    app = create_dash_app(volume)
    app.run(debug=True, port=8050)
