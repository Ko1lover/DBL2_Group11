import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_leaflet as dl
import pandas as pd
from dash import dash_table
PAS_borough = pd.read_csv('assets/PAS_borough_fin.csv')
PAS_borough = PAS_borough.drop(columns=['Date'])
PAS_borough = PAS_borough.groupby('Borough').agg('mean').reset_index()
crime_neighbourhood = pd.read_csv('assets/metropolitan-street-latest.csv')
crime_neighbourhood = crime_neighbourhood.drop(columns=['Longitude','Latitude','month','year','borough'])


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
    dash_table.DataTable(
        id = 'selected-data-borough',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    ),
    dash_table.DataTable(
        id = 'selected-data-neighborhood',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    ),
])

@app.callback(
    Output('selected-data-borough', 'columns'),
    Output('selected-data-borough', 'data'),
    [Input('borough-geojson', 'clickData')]
)
def update_selected_data(borough_feature):
    if borough_feature is not None:
        borough = borough_feature['properties'].get('name', 'N/A')
        borough = PAS_borough[PAS_borough['Borough'] == borough]
        if borough_feature is not None:
            return [{"name": i, "id": i} for i in borough.columns],borough.to_dict('records')
        else:
            return None,None
    return None,None

@app.callback(
    Output('selected-data-neighborhood', 'columns'),
    Output('selected-data-neighborhood', 'data'),
    [Input('neighborhood-geojson', 'clickData')]
)
def update_selected_data(neighborhood_feature):
    if neighborhood_feature is not None:
        neighborhood_name = neighborhood_feature['properties'].get('name', 'N/A')
        print(type(crime_neighbourhood['neighbourhood']))
        neighborhood = crime_neighbourhood[crime_neighbourhood['neighbourhood'] == neighborhood_name]
        if neighborhood is not None:
            return [{"name": i, "id": i} for i in neighborhood.columns], neighborhood.to_dict('records')
        else:
            return None, None
    return None, None

if __name__ == '__main__':
    app.run_server(debug=True)

