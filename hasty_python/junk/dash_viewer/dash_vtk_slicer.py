import dash
from dash import html, dcc, Input, Output, State
import dash_vtk
import numpy as np


def create_vtk_slicer(volume):
    """Create a Dash VTK slicer application."""
    app = dash.Dash(__name__)
    
    ndim = volume.ndim
    shape = volume.shape
    
    # Initialize slice positions
    i_slice = shape[0] // 2 if ndim >= 3 else 0
    j_slice = shape[1] // 2 if ndim >= 3 else 0
    k_slice = shape[2] // 2 if ndim >= 3 else 0
    
    # Stores for current indices
    stores = [
        dcc.Store(id='store-i', data=i_slice),
        dcc.Store(id='store-j', data=j_slice),
        dcc.Store(id='store-k', data=k_slice),
        dcc.Store(id='store-t', data=0),
        dcc.Store(id='store-c', data=0),
        dcc.Store(id='volume-state', data=None),
    ]
    
    # Build controls
    controls = []
    
    # Position labels
    if ndim >= 3:
        controls.append(html.Div([
            html.Label(f'Sagittal (I): {i_slice}', id='label-i', style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Label(f'Coronal (J): {j_slice}', id='label-j', style={'display': 'inline-block', 'margin-right': '20px'}),
            html.Label(f'Axial (K): {k_slice}', id='label-k', style={'display': 'inline-block'}),
        ], style={'margin': '10px', 'textAlign': 'center', 'fontSize': '16px', 'fontWeight': 'bold'}))
    
    # Only add sliders for 4th and 5th dimensions
    if ndim >= 4:
        controls.append(html.Div([
            html.Label(f'Time (T): 0 / {shape[3]-1}', id='label-t'),
            dcc.Slider(
                id='t-slider',
                min=0,
                max=shape[3]-1,
                value=0,
                marks={i: str(i) for i in range(0, shape[3], max(1, shape[3]//5))},
                step=1,
            )
        ], style={'margin': '10px'}))
    
    if ndim >= 5:
        controls.append(html.Div([
            html.Label(f'Channel (C): 0 / {shape[4]-1}', id='label-c'),
            dcc.Slider(
                id='c-slider',
                min=0,
                max=shape[4]-1,
                value=0,
                marks={i: str(i) for i in range(0, shape[4], max(1, shape[4]//5))},
                step=1,
            )
        ], style={'margin': '10px'}))
    
    # Prepare initial volume data
    if ndim == 3:
        vol_3d = volume
    elif ndim == 4:
        vol_3d = volume[:, :, :, 0]
    elif ndim == 5:
        vol_3d = volume[:, :, :, 0, 0]
    else:
        vol_3d = volume
    
    # Ensure float32 and Fortran order
    vol_3d = np.asfortranarray(vol_3d.astype(np.float32))
    
    # Normalize to 0-255 for better visualization
    vol_min, vol_max = vol_3d.min(), vol_3d.max()
    if vol_max > vol_min:
        vol_3d_normalized = ((vol_3d - vol_min) / (vol_max - vol_min) * 255).astype(np.uint8)
    else:
        vol_3d_normalized = np.zeros_like(vol_3d, dtype=np.uint8)
    
    # Flatten in Fortran order
    initial_values = vol_3d_normalized.ravel(order='F').tolist()
    
    # Create three separate VTK views - one for each orientation
    vtk_view = html.Div([
        html.Div([
            # Sagittal view (I slice - YZ plane)
            html.Div([
                html.H3('Sagittal', style={'textAlign': 'center', 'margin': '0'}),
                dash_vtk.View(
                    id='vtk-view-sagittal',
                    children=[
                        dash_vtk.ShareDataSet([
                            dash_vtk.ImageData(
                                id='image-data-sag',
                                dimensions=list(vol_3d.shape),
                                spacing=[1, 1, 1],
                                origin=[0, 0, 0],
                                children=[
                                    dash_vtk.PointData([
                                        dash_vtk.DataArray(
                                            registration='setScalars',
                                            values=initial_values,
                                        )
                                    ])
                                ],
                            ),
                        ]),
                        dash_vtk.SliceRepresentation(
                            id='slice-i',
                            iSlice=i_slice,
                            property={'colorWindow': 255.0, 'colorLevel': 127.5},
                            children=[dash_vtk.ShareDataSet()],
                        ),
                    ],
                    style={'width': '100%', 'height': '450px'},
                    cameraPosition=[vol_3d.shape[0] * 2, vol_3d.shape[1] / 2, vol_3d.shape[2] / 2],
                    cameraViewUp=[0, 0, 1],
                    cameraParallelProjection=True,
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Coronal view (J slice - XZ plane)
            html.Div([
                html.H3('Coronal', style={'textAlign': 'center', 'margin': '0'}),
                dash_vtk.View(
                    id='vtk-view-coronal',
                    children=[
                        dash_vtk.ShareDataSet([
                            dash_vtk.ImageData(
                                id='image-data-cor',
                                dimensions=list(vol_3d.shape),
                                spacing=[1, 1, 1],
                                origin=[0, 0, 0],
                                children=[
                                    dash_vtk.PointData([
                                        dash_vtk.DataArray(
                                            registration='setScalars',
                                            values=initial_values,
                                        )
                                    ])
                                ],
                            ),
                        ]),
                        dash_vtk.SliceRepresentation(
                            id='slice-j',
                            jSlice=j_slice,
                            property={'colorWindow': 255.0, 'colorLevel': 127.5},
                            children=[dash_vtk.ShareDataSet()],
                        ),
                    ],
                    style={'width': '100%', 'height': '450px'},
                    cameraPosition=[vol_3d.shape[0] / 2, vol_3d.shape[1] * 2, vol_3d.shape[2] / 2],
                    cameraViewUp=[0, 0, 1],
                    cameraParallelProjection=True,
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Axial view (K slice - XY plane)
            html.Div([
                html.H3('Axial', style={'textAlign': 'center', 'margin': '0'}),
                dash_vtk.View(
                    id='vtk-view-axial',
                    children=[
                        dash_vtk.ShareDataSet([
                            dash_vtk.ImageData(
                                id='image-data-ax',
                                dimensions=list(vol_3d.shape),
                                spacing=[1, 1, 1],
                                origin=[0, 0, 0],
                                children=[
                                    dash_vtk.PointData([
                                        dash_vtk.DataArray(
                                            registration='setScalars',
                                            values=initial_values,
                                        )
                                    ])
                                ],
                            ),
                        ]),
                        dash_vtk.SliceRepresentation(
                            id='slice-k',
                            kSlice=k_slice,
                            property={'colorWindow': 255.0, 'colorLevel': 127.5},
                            children=[dash_vtk.ShareDataSet()],
                        ),
                    ],
                    style={'width': '100%', 'height': '450px'},
                    cameraPosition=[vol_3d.shape[0] / 2, vol_3d.shape[1] / 2, vol_3d.shape[2] * 2],
                    cameraViewUp=[0, 1, 0],
                    cameraParallelProjection=True,
                ),
            ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '100%'}),
    ], style={'width': '100%'})
    
    app.layout = html.Div([
        html.H1('Medical Image Slicer (VTK)', style={'textAlign': 'center'}),
        html.Div('Click and drag on slices to navigate through the volume', 
                 style={'textAlign': 'center', 'color': '#666', 'marginBottom': '10px'}),
        html.Div(controls, style={'padding': '20px'}),
        html.Div(stores),
        vtk_view,
    ])
    
    # Update volume data when time/channel changes
    @app.callback(
        [Output('image-data-sag', 'children'),
         Output('image-data-sag', 'dimensions'),
         Output('image-data-cor', 'children'),
         Output('image-data-cor', 'dimensions'),
         Output('image-data-ax', 'children'),
         Output('image-data-ax', 'dimensions')],
        [Input('store-t', 'data'),
         Input('store-c', 'data')],
    )
    def update_volume_data(t_idx, c_idx):
        # Get the appropriate 3D slice
        if ndim == 3:
            vol_3d = volume
        elif ndim == 4:
            vol_3d = volume[:, :, :, t_idx if t_idx is not None else 0]
        elif ndim == 5:
            vol_3d = volume[:, :, :, t_idx if t_idx is not None else 0, c_idx if c_idx is not None else 0]
        else:
            vol_3d = volume
        
        # Ensure float32 and Fortran order
        vol_3d = np.asfortranarray(vol_3d.astype(np.float32))
        
        # Normalize to 0-255 for better visualization
        vol_min, vol_max = vol_3d.min(), vol_3d.max()
        if vol_max > vol_min:
            vol_3d = ((vol_3d - vol_min) / (vol_max - vol_min) * 255).astype(np.uint8)
        else:
            vol_3d = np.zeros_like(vol_3d, dtype=np.uint8)
        
        # Flatten in Fortran order and convert to list for JSON serialization
        values_list = vol_3d.ravel(order='F').tolist()
        
        # Return children and dimensions (same for all three views)
        children = [
            dash_vtk.PointData([
                dash_vtk.DataArray(
                    registration='setScalars',
                    values=values_list,
                )
            ])
        ]
        dims = list(vol_3d.shape)
        
        return children, dims, children, dims, children, dims
    
    # Update slice positions
    @app.callback(
        [Output('slice-i', 'iSlice'),
         Output('slice-j', 'jSlice'),
         Output('slice-k', 'kSlice'),
         Output('slice-i', 'property'),
         Output('slice-j', 'property'),
         Output('slice-k', 'property')],
        [Input('store-i', 'data'),
         Input('store-j', 'data'),
         Input('store-k', 'data')],
    )
    def update_slice_positions(i_val, j_val, k_val):
        prop = {'colorWindow': 255.0, 'colorLevel': 127.5}
        return i_val, j_val, k_val, prop, prop, prop
    
    # Update slice indices from sliders (4th and 5th dimensions only)
    if ndim >= 4:
        @app.callback(
            Output('store-t', 'data'),
            Input('t-slider', 'value'),
        )
        def update_t(val):
            return val
    
    if ndim >= 5:
        @app.callback(
            Output('store-c', 'data'),
            Input('c-slider', 'value'),
        )
        def update_c(val):
            return val
    
    # Update labels
    @app.callback(
        [Output('label-i', 'children'),
         Output('label-j', 'children'),
         Output('label-k', 'children')] +
        ([Output('label-t', 'children')] if ndim >= 4 else []) +
        ([Output('label-c', 'children')] if ndim >= 5 else []),
        [Input('store-i', 'data'),
         Input('store-j', 'data'),
         Input('store-k', 'data'),
         Input('store-t', 'data'),
         Input('store-c', 'data')],
    )
    def update_labels(i_val, j_val, k_val, t_val, c_val):
        outputs = [
            f'Sagittal (I): {i_val}',
            f'Coronal (J): {j_val}',
            f'Axial (K): {k_val}',
        ]
        if ndim >= 4:
            outputs.append(f'Time (T): {t_val} / {shape[3]-1}')
        if ndim >= 5:
            outputs.append(f'Channel (C): {c_val} / {shape[4]-1}')
        return outputs
    
    return app


# Example usage
if __name__ == '__main__':
    # Create sample 3D volume
    volume = np.random.rand(100, 100, 100).astype(np.float32)
    
    # For 4D: volume = np.random.rand(100, 100, 100, 20).astype(np.float32)
    # For 5D: volume = np.random.rand(100, 100, 100, 20, 3).astype(np.float32)
    
    app = create_vtk_slicer(volume)
    app.run(debug=True, port=8051)
