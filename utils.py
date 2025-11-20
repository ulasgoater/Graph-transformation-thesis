# utils.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import math

def header_cols(path):
    return list(pd.read_csv(path, nrows=0).columns)

def get_utm_epsg_from_bbox(bbox):
    """Calculate the appropriate UTM EPSG code from bounding box.
    
    Args:
        bbox: [south, west, north, east] in decimal degrees
    
    Returns:
        int: EPSG code for the appropriate UTM zone
    """
    south, west, north, east = bbox
    # Use center of bbox
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    
    # Calculate UTM zone (1-60)
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Determine hemisphere: Northern (326XX) or Southern (327XX)
    if center_lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    
    return epsg_code

def ensure_meters(gdf, target_epsg=None):
    """Convert GeoDataFrame to metric CRS.
    
    Args:
        gdf: GeoDataFrame to convert
        target_epsg: Specific EPSG code to use, or None to use Web Mercator
    
    Returns:
        GeoDataFrame in metric projection
    """
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    if target_epsg:
        target_crs = f"EPSG:{target_epsg}"
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(epsg=target_epsg)
    else:
        # Fallback to Web Mercator
        if gdf.crs.to_string() != "EPSG:3857":
            gdf = gdf.to_crs(epsg=3857)
    
    return gdf

def snap_coord_tuple(x, y, tol):
    # deterministic snap by rounding to multiples of tol
    # returns tuple of floats
    rx = round(round(x / tol) * tol, 6)
    ry = round(round(y / tol) * tol, 6)
    return (rx, ry)

def geom_midpoint_on_line(line, frac=0.5):
    # returns a point at fraction of length
    if line is None or line.length == 0:
        return None
    return line.interpolate(frac * line.length)

def longest_linestring_from_multigeom(geom):
    # If MultiLineString, return longest part, otherwise return geom
    try:
        from shapely.geometry import MultiLineString
        if isinstance(geom, MultiLineString):
            parts = list(geom)
            parts.sort(key=lambda p: p.length, reverse=True)
            return parts[0]
    except Exception:
        pass
    return geom
