# io_utils.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from utils import ensure_meters, header_cols
import re

def load_autograph_csv(path):
    """
    Load autograph CSV and build LineString geometries.
    Supports two formats:
    1. GeoJSON in 'geometry' column (Milano format)
    2. geometry/{i}/lat & geometry/{i}/lon columns (Bologna format)
    Returns GeoDataFrame in WORK_CRS.
    """
    df = pd.read_csv(path, low_memory=False)
    
    # Check if we have GeoJSON geometry column
    if 'geometry' in df.columns:
        try:
            # Try to load directly with geopandas
            gdf = gpd.read_file(path)
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                gdf = ensure_meters(gdf)
                return gdf
        except Exception:
            pass
        
        # Fallback: parse GeoJSON strings manually
        import json
        def parse_geojson(geom_str):
            if pd.isna(geom_str):
                return None
            try:
                geojson = json.loads(geom_str)
                if geojson['type'] == 'LineString':
                    coords = geojson['coordinates']
                    return LineString(coords)
            except Exception:
                return None
        
        df['geometry'] = df['geometry'].apply(parse_geojson)
        df = df[df['geometry'].notna()].copy()
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        gdf = ensure_meters(gdf)
        return gdf
    
    # Fallback to old Bologna format with geometry/*/lat columns
    # detect geometry/*/lat columns
    geom_lat_cols = [c for c in df.columns if re.match(r"geometry/\d+/lat", c)]
    geom_idx = sorted({int(c.split("/")[1]) for c in geom_lat_cols}) if geom_lat_cols else []
    if not geom_idx:
        raise RuntimeError("Autograph CSV: no geometry columns detected (neither GeoJSON nor lat/lon format).")

    def make_linestring(row):
        pts = []
        for i in geom_idx:
            lat_k, lon_k = f"geometry/{i}/lat", f"geometry/{i}/lon"
            if lat_k in row and lon_k in row:
                lat = row[lat_k]; lon = row[lon_k]
                if pd.notna(lat) and pd.notna(lon):
                    try:
                        pts.append((float(lon), float(lat)))
                    except Exception:
                        continue
        return LineString(pts) if len(pts) >= 2 else None

    df["geometry"] = df.apply(make_linestring, axis=1)
    df = df[df["geometry"].notna()].copy()
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = ensure_meters(gdf)
    return gdf

def load_point_csv(path, lon_candidates=("lon","longitude"), lat_candidates=("lat","latitude","center_lon","center/long","center/lat","center_lon")):
    df = pd.read_csv(path, low_memory=False)
    
    # Check if we have GeoJSON geometry column
    if 'geometry' in df.columns:
        try:
            # Try to load directly with geopandas
            gdf = gpd.read_file(path)
            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                gdf = ensure_meters(gdf)
                return gdf
        except Exception:
            pass
        
        # Fallback: parse GeoJSON strings manually
        import json
        def parse_geojson_point(geom_str):
            if pd.isna(geom_str):
                return None
            try:
                geojson = json.loads(geom_str)
                if geojson['type'] == 'Point':
                    coords = geojson['coordinates']
                    return Point(coords)
            except Exception:
                return None
        
        df['geometry'] = df['geometry'].apply(parse_geojson_point)
        df = df[df['geometry'].notna()].copy()
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        gdf = ensure_meters(gdf)
        return gdf
    
    # Fallback to lat/lon columns
    lon_col = next((c for c in df.columns if c.lower() in [x.lower() for x in lon_candidates]), None)
    lat_col = next((c for c in df.columns if c.lower() in [x.lower() for x in lat_candidates]), None)
    if lon_col is None or lat_col is None:
        # try center/lat & center/lon style
        lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    if lon_col is None or lat_col is None:
        raise RuntimeError(f"No geometry or lon/lat columns found in {path}")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    gdf = ensure_meters(gdf)
    return gdf
