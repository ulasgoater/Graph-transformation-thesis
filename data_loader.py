"""Data Loading Module

Purpose: Read raw CSV exports (roads, pedestrian ways, amenities, crossings) and
convert them into GeoDataFrames in a consistent projected CRS (EPSG:3857) so
downstream steps can measure distances in meters.

Beginner concepts:
1. GeoDataFrame: Like a pandas DataFrame but each row has a geometry shape.
2. CRS (Coordinate Reference System): Defines how coordinates map to Earth.
    - Input OSM-like data: usually latitude/longitude (EPSG:4326).
    - We project to EPSG:3857 so buffer distances are in meters.
3. Filtering: We optionally filter pedestrian features to avoid labeling every
    road as positive in dense datasets.

High-level flow:
  load_gdfs() -> (roads, pedestrian, amenities, crossings)
     |_ _process_* helper functions each normalize columns & build geometry
"""

import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from config import (
    AUTOGRAPH_CSV, PEDESTRIAN_CSV, AMENITIES_CSV, CROSSINGS_CSV,
    PEDESTRIAN_ALLOWED_HIGHWAY, PEDESTRIAN_ALLOWED_FOOTWAY
)
from utils import (
    header_cols, safe_rename, ensure_meters, make_linestring_for_row, try_parse_geom
)


def load_gdfs(autograph_csv=AUTOGRAPH_CSV, pedestrian_csv=PEDESTRIAN_CSV,
              amenities_csv=AMENITIES_CSV, crossings_csv=CROSSINGS_CSV):
    """
    Load and process all CSV files into GeoDataFrames.
    
    Returns:
        tuple: (gdf_auto, gdf_ped, gdf_amen, gdf_cross) - all in EPSG:3857
    """
    # 1. Read ONLY headers first so we can decide which columns to load (saves memory/time)
    auto_h = header_cols(autograph_csv)
    ped_h = header_cols(pedestrian_csv)
    amen_h = header_cols(amenities_csv)
    cross_h = header_cols(crossings_csv)

    # 2. Discover geometry coordinate columns in autograph file: they appear as geometry/0/lat, geometry/0/lon, etc.
    geom_lat_cols = [c for c in auto_h if re.match(r"geometry/\d+/lat", c)]
    geom_indices = sorted({int(c.split("/")[1]) for c in geom_lat_cols}) if geom_lat_cols else []
    geom_cols = []
    for i in geom_indices:
        lat_k, lon_k = f"geometry/{i}/lat", f"geometry/{i}/lon"
        if lat_k in auto_h and lon_k in auto_h:
            geom_cols += [lat_k, lon_k]

    # 3. Enumerate which tag/attribute columns we care about (avoid loading entire huge CSVs).
    desired_auto_tags = [
        "tags/highway", "tags/sidewalk", "tags/sidewalk:left", "tags/sidewalk:right",
        "tags/maxspeed", "tags/lanes", "tags/living_street", "tags/surface", 
        "tags/lit", "tags/oneway", "tags/sidewalk:both"
    ]
    auto_usecols = [c for c in desired_auto_tags if c in auto_h] + geom_cols

    ped_pref = [
        "geometry", "coordinates", "@id", "latitude", "longitude", "lat", "lon",
        "highway", "sidewalk", "footway", "surface", "tactile_paving", 
        "crossing", "kerb", "lit"
    ]
    ped_usecols = [c for c in ped_pref if c in ped_h]

    amen_pref = [
        "type", "id", "lat", "lon", "center/lat", "center/lon", "tags/amenity",
        "tags/shop", "tags/healthcare", "tags/school", "tags/tourism", 
        "tags/leisure", "tags/wheelchair", "tags/name"
    ]
    amen_usecols = [c for c in amen_pref if c in amen_h]

    cross_pref = [
        "type", "id", "lat", "lon", "tags/crossing", "tags/crossing:island",
        "tags/tactile_paving", "tags/kerb", "tags/traffic_signals", 
        "tags/button_operated", "tags/lit", "tags/surface"
    ]
    cross_usecols = [c for c in cross_pref if c in cross_h]

    # 4. Load chosen columns from each CSV
    df_auto = pd.read_csv(autograph_csv, usecols=auto_usecols, low_memory=False)
    df_ped = pd.read_csv(pedestrian_csv, usecols=ped_usecols, low_memory=False)
    df_amen = pd.read_csv(amenities_csv, usecols=amen_usecols, low_memory=False)
    df_cross = pd.read_csv(crossings_csv, usecols=cross_usecols, low_memory=False)

    # 5. Process & normalize each dataset into GeoDataFrames with projected CRS
    gdf_auto = _process_autograph_data(df_auto, geom_indices)
    gdf_ped = _process_pedestrian_data(df_ped)

    # Optional filtering to focus on meaningful pedestrian geometries for labeling
    gdf_ped = _filter_pedestrian(gdf_ped)
    gdf_amen = _process_amenities_data(df_amen)
    gdf_cross = _process_crossings_data(df_cross)

    print("Loaded:", len(gdf_auto), "auto;", len(gdf_ped), "ped;", 
        len(gdf_amen), "amenities;", len(gdf_cross), "crossings")  # quick sanity counts
    
    return gdf_auto, gdf_ped, gdf_amen, gdf_cross


def _process_autograph_data(df_auto, geom_indices):
    """Convert the road (autograph) CSV into a projected GeoDataFrame.

    Steps:
    1. Rename raw tag columns -> simple snake_case names.
    2. Build LineString geometry from sequential point columns.
    3. Filter out rows with invalid or <2-point geometry.
    4. Project to EPSG:3857.
    """
    # Normalize column names
    df_auto = safe_rename(df_auto, {
        "tags/highway": "tags_highway",
        "tags/sidewalk": "tags_sidewalk",
        "tags/sidewalk:left": "tags_sidewalk_left",
        "tags/sidewalk:right": "tags_sidewalk_right",
        "tags/maxspeed": "tags_maxspeed",
        "tags/lanes": "tags_lanes",
        "tags/lit": "tags_lit",
        "tags/surface": "tags_surface",
        "tags/oneway": "tags_oneway",
        "tags/sidewalk:both": "tags_sidewalk_both",
        "tags/living_street": "tags_living_street"
    })

    # Create LineString geometry from coordinate columns
    if not geom_indices:
        raise RuntimeError("Autograph CSV: no geometry/*/lat columns found.")

    df_auto = df_auto.copy()
    df_auto["geometry"] = df_auto.apply(
        lambda r: make_linestring_for_row(r, geom_indices), axis=1
    )
    df_auto = df_auto[df_auto["geometry"].notna()].copy()
    
    gdf_auto = gpd.GeoDataFrame(df_auto, geometry="geometry", crs="EPSG:4326")
    return ensure_meters(gdf_auto)


def _process_pedestrian_data(df_ped):
    """Create pedestrian GeoDataFrame (points or parsed WKT lines).

    We attempt WKT geometry parsing first (if present); otherwise we fall back
    to latitude/longitude columns.
    """
    df_ped = safe_rename(df_ped, {
        "latitude": "latitude",
        "longitude": "longitude",
        "lat": "latitude",
        "lon": "longitude"
    })

    # Try WKT geometry first, then coordinates
    if "geometry" in df_ped.columns and df_ped["geometry"].notna().any():
        df_ped["geom_try"] = df_ped["geometry"].apply(try_parse_geom)
        if df_ped["geom_try"].notna().any():
            gdf_ped = gpd.GeoDataFrame(df_ped, geometry="geom_try", crs="EPSG:4326")
        elif {"longitude", "latitude"}.issubset(df_ped.columns) or {"lon", "lat"}.issubset(df_ped.columns):
            lonc = "longitude" if "longitude" in df_ped.columns else "lon"
            latc = "latitude" if "latitude" in df_ped.columns else "lat"
            gdf_ped = gpd.GeoDataFrame(
                df_ped, geometry=gpd.points_from_xy(df_ped[lonc], df_ped[latc]), crs="EPSG:4326"
            )
        else:
            raise RuntimeError("Pedestrian CSV: could not parse geometry and no lon/lat found.")
    else:
        lonc = "longitude" if "longitude" in df_ped.columns else ("lon" if "lon" in df_ped.columns else None)
        latc = "latitude" if "latitude" in df_ped.columns else ("lat" if "lat" in df_ped.columns else None)
        if lonc and latc:
            gdf_ped = gpd.GeoDataFrame(
                df_ped, geometry=gpd.points_from_xy(df_ped[lonc], df_ped[latc]), crs="EPSG:4326"
            )
        else:
            raise RuntimeError("Pedestrian CSV: no lon/lat or usable geometry found.")

    return ensure_meters(gdf_ped)


def _filter_pedestrian(gdf_ped):
    """Optionally reduce pedestrian features to specific tag subsets.

    Why: If every footway/path is included everywhere, many roads might end up
    extremely close to some pedestrian geometry, making everything positive.
    Filtering keeps the classification task meaningful.
    """
    original_len = len(gdf_ped)
    try:
        if PEDESTRIAN_ALLOWED_HIGHWAY and "highway" in gdf_ped.columns:
            gdf_ped = gdf_ped[gdf_ped["highway"].isin(PEDESTRIAN_ALLOWED_HIGHWAY)].copy()
        if PEDESTRIAN_ALLOWED_FOOTWAY and "footway" in gdf_ped.columns:
            gdf_ped = gdf_ped[gdf_ped["footway"].isin(PEDESTRIAN_ALLOWED_FOOTWAY)].copy()
    except Exception:
        # Fail safe: return unfiltered if something unexpected occurs
        return gdf_ped
    if len(gdf_ped) != original_len:
        print(f"Filtered pedestrian features: {original_len} -> {len(gdf_ped)} after tag constraints")
    return gdf_ped


def _process_amenities_data(df_amen):
    """Convert amenities CSV to GeoDataFrame choosing the best available coordinate columns."""
    df_amen = safe_rename(df_amen, {
        "center/lat": "center_lat",
        "center/lon": "center_lon",
        "tags/amenity": "tags_amenity",
        "tags/shop": "tags_shop",
        "tags/healthcare": "tags_healthcare",
        "tags/school": "tags_school",
        "tags/leisure": "tags_leisure",
        "tags/name": "tags_name"
    })

    # Create geometry from coordinates
    if {"center_lat", "center_lon"}.issubset(df_amen.columns):
        gdf_amen = gpd.GeoDataFrame(
            df_amen, geometry=gpd.points_from_xy(df_amen["center_lon"], df_amen["center_lat"]), crs="EPSG:4326"
        )
    elif {"lon", "lat"}.issubset(df_amen.columns):
        gdf_amen = gpd.GeoDataFrame(
            df_amen, geometry=gpd.points_from_xy(df_amen["lon"], df_amen["lat"]), crs="EPSG:4326"
        )
    else:
        lon_cols = [c for c in df_amen.columns if c.lower().endswith("lon") or c.lower().endswith("longitude")]
        lat_cols = [c for c in df_amen.columns if c.lower().endswith("lat") or c.lower().endswith("latitude")]
        if lon_cols and lat_cols:
            gdf_amen = gpd.GeoDataFrame(
                df_amen, geometry=gpd.points_from_xy(df_amen[lon_cols[0]], df_amen[lat_cols[0]]), crs="EPSG:4326"
            )
        else:
            raise RuntimeError("Amenities CSV: no coordinate columns found.")

    return ensure_meters(gdf_amen)


def _process_crossings_data(df_cross):
    """Convert crossings CSV to GeoDataFrame and normalize tag columns."""
    df_cross = safe_rename(df_cross, {
        "tags/crossing": "tags_crossing",
        "tags/crossing:island": "tags_crossing_island",
        "tags/tactile_paving": "tags_tactile_paving",
        "tags/kerb": "tags_kerb",
        "tags/traffic_signals": "tags_traffic_signals",
        "tags/button_operated": "tags_button_operated",
        "tags/lit": "tags_lit"
    })

    # Create geometry from coordinates
    if {"lon", "lat"}.issubset(df_cross.columns):
        gdf_cross = gpd.GeoDataFrame(
            df_cross, geometry=gpd.points_from_xy(df_cross["lon"], df_cross["lat"]), crs="EPSG:4326"
        )
    elif {"longitude", "latitude"}.issubset(df_cross.columns):
        gdf_cross = gpd.GeoDataFrame(
            df_cross, geometry=gpd.points_from_xy(df_cross["longitude"], df_cross["latitude"]), crs="EPSG:4326"
        )
    else:
        raise RuntimeError("Crossings CSV: no lon/lat columns found.")

    return ensure_meters(gdf_cross)