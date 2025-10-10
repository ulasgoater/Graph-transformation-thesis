"""General Utility Helpers

Small, sharply focused helper functions used across multiple modules. Keeping
them here avoids circular imports and repetition. Each function is intentionally
lightweight and side‑effect free (pure) where feasible to simplify testing and
reasoning.
"""

import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely import wkt


def header_cols(path):
    """Return list of column headers from a CSV without reading full data.

    Optimization: Reads only the first (header) row, saving memory for large files.
    """
    return list(pd.read_csv(path, nrows=0).columns)


def safe_rename(df, mapping):
    """Rename only columns that actually exist (ignores missing keys)."""
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    if existing:
        return df.rename(columns=existing)
    return df


def ensure_meters(gdf):
    """Project GeoDataFrame to Web Mercator (EPSG:3857) if needed.

    Rationale: Distance / buffer operations in the pipeline assume planar meters.
    If CRS is missing we assume WGS84 (EPSG:4326) then project.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)
    return gdf


def make_linestring_for_row(row, geom_indices):
    """Build a LineString from repeated geometry/<i>/(lat|lon) columns.

    Skips malformed or incomplete coordinate pairs and returns None if fewer
    than two valid points exist (cannot form a LineString).
    """
    pts = []
    for i in geom_indices:
        lat_k, lon_k = f"geometry/{i}/lat", f"geometry/{i}/lon"
        if lat_k in row.index and lon_k in row.index:
            lat = row[lat_k]
            lon = row[lon_k]
            if pd.notna(lat) and pd.notna(lon):
                try:
                    pts.append((float(lon), float(lat)))
                except Exception:
                    continue
    return LineString(pts) if len(pts) >= 2 else None


def try_parse_geom(s):
    """Best‑effort parse of a WKT geometry string to a Shapely object.

    Returns None on failure instead of raising to keep ingestion robust.
    """
    if pd.isna(s):
        return None
    if isinstance(s, str) and (
        s.strip().upper().startswith("POINT") or 
        s.strip().upper().startswith("LINESTRING") or 
        s.strip().upper().startswith("MULTILINESTRING")
    ):
        try:
            return wkt.loads(s)
        except Exception:
            return None
    return None