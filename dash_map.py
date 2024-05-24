import dash
from dash import html
from dash.dependencies import Input, Output
import dash_leaflet as dl

app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("London Interactive Map"),
    dl.Map(
        id='map',
        center=[51.5074, -0.1278],
        zoom=10,
        style={'width': '100%', 'height': '80vh'},
        children=[
            dl.TileLayer(),
            dl.LayersControl(
                [
                    dl.BaseLayer(dl.TileLayer(), name='Base', checked=True),
                    dl.Overlay(
                        dl.GeoJSON(url='assets/Boroughs_boundaries.geojson', id='borough-geojson',
                                   options=dict(style=dict(color='blue', weight=2))),
                        name='Borough',
                        checked=True
                    ),
                    dl.Overlay(
                        dl.GeoJSON(url='assets/neighborhood_boundaries.geojson', id='neighborhood-geojson',
                                   options=dict(style=dict(color='green', weight=1))),
                        name='Neighborhood',
                        checked=False
                    )
                ]
            )
        ]
    ),
    html.Div(id='selected-data-borough'),
    html.Div(id='selected-data-neighborhood')
])

@app.callback(
    Output('selected-data-borough', 'children'),
    [Input('borough-geojson', 'clickData')]
)
def update_selected_data(borough_feature):
    if borough_feature is not None:
        borough = borough_feature['properties'].get('name', 'N/A')
        return f"You selected borough '{borough}'."
    return "No region selected."

@app.callback(
    Output('selected-data-neighborhood', 'children'),
    [Input('neighborhood-geojson', 'clickData')]
)
def update_selected_data(neighborhood_feature):
    if neighborhood_feature is not None:
        neighborhood_feature = neighborhood_feature['properties'].get('name', 'N/A')
        return f"You selected '{neighborhood_feature}'."
    return "No region selected."

if __name__ == '__main__':
    app.run_server(debug=True)
