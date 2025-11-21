"""Utilities to fetch Milano OSM data directly from the Overpass API."""
from __future__ import annotations

import itertools
import time
from typing import Dict, Iterable, Optional

import geopandas as gpd
import overpass
from shapely.geometry import shape

from config import OVERPASS_QUERIES
from utils import ensure_meters, get_utm_epsg_from_bbox


class OSMDataFetcher:
    """Wraps the `overpass` client and converts responses to GeoDataFrames."""

    def __init__(self, timeout: int = 60, max_retries: int = 3, retry_delay: float = 5.0):
        self.api = overpass.API(timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @staticmethod
    def _format_bbox(bbox: Iterable[float]) -> str:
        south, west, north, east = bbox
        return f"{south},{west},{north},{east}"

    def fetch(self, query_key: str, bbox: Iterable[float]) -> gpd.GeoDataFrame:
        if query_key not in OVERPASS_QUERIES:
            raise KeyError(f"Unknown OVERPASS query key: {query_key}")
        
        if not bbox:
            raise ValueError("bbox parameter is required")
        
        bbox = list(bbox)
        query = OVERPASS_QUERIES[query_key].format(bbox=self._format_bbox(bbox))
        
        # Calculate appropriate UTM zone for this bbox
        target_epsg = get_utm_epsg_from_bbox(bbox)

        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.api.get(query, verbosity="geom")
                break
            except overpass.errors.ServerLoadError:
                if attempt == self.max_retries:
                    raise
                delay = self.retry_delay * attempt
                print(f"Overpass server busy, retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries})")
                time.sleep(delay)

        if response is None:
            raise RuntimeError("Overpass request failed with no response")
        features = response.get("features", [])
        if not features:
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        # Ensure we always have a geometry column even if Overpass returns feature properties only.
        if "geometry" not in gdf.columns:
            gdf["geometry"] = [shape(feat["geometry"]) for feat in features]
            gdf.set_geometry("geometry", inplace=True)

        return ensure_meters(gdf, target_epsg=target_epsg)


def fetch_autograph_data(bbox: Iterable[float]) -> gpd.GeoDataFrame:
    return OSMDataFetcher().fetch("autograph", bbox)


def fetch_crossings_data(bbox: Iterable[float]) -> gpd.GeoDataFrame:
    gdf = OSMDataFetcher().fetch("crossings", bbox)
    if gdf.empty:
        return gdf
    if not gdf.geom_type.isin(["Point"]).all():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.centroid
    return gdf


def fetch_pedestrian_data(bbox: Iterable[float]) -> gpd.GeoDataFrame:
    return OSMDataFetcher().fetch("pedestrian", bbox)


def fetch_amenities_data(bbox: Iterable[float]) -> gpd.GeoDataFrame:
    return OSMDataFetcher().fetch("amenities", bbox)
