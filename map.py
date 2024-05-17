import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point


df_accidents = pd.read_csv('metropolitan-street.csv')

df_accidents = df_accidents.dropna(subset=['Latitude', 'Longitude'])
geometry_accidents = [Point(xy) for xy in zip(df_accidents['Longitude'], df_accidents['Latitude'])]


gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry_accidents)


gdf_boroughs = gpd.read_file('london_boroughs.gpkg')


m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)


folium.GeoJson(gdf_boroughs).add_to(m)


for idx, row in gdf_accidents.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']]).add_to(m)

m.save('london_accidents_with_boroughs_map.html')
