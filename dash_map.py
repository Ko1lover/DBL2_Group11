import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_leaflet as dl
import pandas as pd
from dash import dash_table

# download boundray file:
# https://data.london.gov.uk/download/london_boroughs/9502cdec-5df0-46e3-8aa1-2b5c5233a31f/London_Boroughs.gpkg

import geopandas as gpd

gdf = gpd.read_file("London_Boroughs.gpkg")

gdf = gdf.to_crs(epsg=4326)

gdf.to_file("Boroughs_boundaries.geojson", driver="GeoJSON")

PAS_borough = pd.read_csv('assets/FINAL_agg_Dataset.csv')
PAS_borough['Date'] = pd.to_datetime(PAS_borough['year'].astype(str) + 'Q' + PAS_borough['quarter'].astype(str))
PAS_borough = PAS_borough.drop(columns=['year', 'quarter'])
PAS_borough = PAS_borough[['Date', 'borough', 'GoodJoblocal', 'TrustMPS',
                           'violent_q_crimes', 'property_q_crimes', 'public_order_q_crimes', 'drug_or_weapon_q_crimes',
                           'other_q_crimes']]
PAS_borough['stop_search_data'] = '...'
PAS_borough['street_crime_data'] = '...'
PAS_borough['PAS_data'] = '...'

quarters = PAS_borough['Date'].dt.to_period('Q').drop_duplicates().sort_values().tolist()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("London Interactive Map"),
    dcc.Dropdown(
        id='quarter-dropdown',
        options=[{'label': str(q), 'value': str(q)} for q in quarters],
        value=str(quarters[0])
    ),
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
                        dl.GeoJSON(url='Boroughs_boundaries.geojson', id='borough-geojson',
                                   zoomToBounds=True,
                                   hoverStyle=dict(weight=5, color='#666', dashArray='')),
                        name='Borough',
                        checked=True,
                    )
                ]
            )
        ]
    ),
    dash_table.DataTable(
        id='selected-data-borough',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    )
])


@app.callback(
    Output('selected-data-borough', 'columns'),
    Output('selected-data-borough', 'data'),
    [Input('borough-geojson', 'clickData'), Input('quarter-dropdown', 'value')]
)
def update_selected_data(borough_feature, selected_quarter):
    if borough_feature is not None and selected_quarter is not None:
        borough = borough_feature['properties'].get('name', 'N/A')
        selected_quarter = pd.Period(selected_quarter, freq='Q')
        borough_data = PAS_borough[
            (PAS_borough['borough'] == borough) & (PAS_borough['Date'].dt.to_period('Q') == selected_quarter)]
        if not borough_data.empty:
            return [{"name": i, "id": i} for i in borough_data.columns], borough_data.to_dict('records')
    return [], []


if __name__ == '__main__':
    app.run_server(debug=True)
