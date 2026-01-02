import requests
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def fetch_boundaries(continent="Americas", region="South America", limit=100):
    """
    Fetches administrative boundaries from the public OpenDataSoft API for a specified continent and region.
    
    Parameters:
        continent (str): The continent to filter the boundaries (by default: "Americas").
        region (str): The region within the continent (by default: "South America").
        limit (int): The number of records to fetch (default: 100). 
        
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the boundaries as geometries.
    """

    url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/records?limit={limit}&refine=continent%3A%22{continent}%22&refine=region%3A%22{region}%22"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        geometries = []

        for record in data.get("results", []):
            geo_shape = record.get("geo_shape", {})
            geometry = geo_shape.get("geometry", {})

            if geometry and geometry["type"] in ["Polygon", "MultiPolygon"]:
                coords = geometry["coordinates"]

                if geometry["type"] == "Polygon":
                    geometries.append(Polygon(coords[0]))  # Convert to Polygon
                elif geometry["type"] == "MultiPolygon":
                    multi_poly = MultiPolygon([Polygon(p[0]) for p in coords])
                    geometries.append(multi_poly)

        return gpd.GeoDataFrame(geometry=geometries)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return gpd.GeoDataFrame()  # Return empty GeoDataFrame in case of failure


