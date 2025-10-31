# utils.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import math

def header_cols(path):
    return list(pd.read_csv(path, nrows=0).columns)

def ensure_meters(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
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
