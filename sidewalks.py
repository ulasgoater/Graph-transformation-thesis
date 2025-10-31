# sidewalks.py
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring
import geopandas as gpd
from utils import longest_linestring_from_multigeom

def make_sidewalks(gdf_roads, offset_m, keep_cols=None):
    """
    For each road segment (LineString), deterministically create two offset lines:
    - left offset (side='left')
    - right offset (side='right')
    
    Line direction is normalized (west-to-east, or south-to-north if same longitude)
    to ensure consistent left/right orientation across all roads.
    
    Returns a GeoDataFrame of sidewalk segments with stable deterministic ordering.
    """
    records = []
    for idx, row in gdf_roads.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        geom = longest_linestring_from_multigeom(geom)
        if geom.length < 0.001:
            continue
        
        # Normalize direction: west-to-east or south-to-north
        coords = list(geom.coords)
        start, end = coords[0], coords[-1]
        # If line goes east-to-west (or south-to-north when same longitude), reverse it
        if start[0] > end[0] or (start[0] == end[0] and start[1] > end[1]):
            geom = LineString(coords[::-1])
        
        # use shapely parallel_offset. It may return LineString or MultiLineString
        try:
            left = geom.parallel_offset(offset_m, 'left', join_style=2)
            right = geom.parallel_offset(offset_m, 'right', join_style=2)
        except Exception:
            # Some geometries may fail; skip deterministically
            continue

        def normalize_line(ls):
            if ls is None:
                return None
            if isinstance(ls, MultiLineString):
                # choose longest piece deterministically
                parts = list(ls.geoms)
                parts.sort(key=lambda p: p.length, reverse=True)
                return parts[0]
            return ls

        left = normalize_line(left)
        right = normalize_line(right)
        # deterministic: create left first then right
        base_props = {}
        if keep_cols:
            for c in keep_cols:
                if c and c in row:
                    base_props[c] = row[c]
        # store
        if left and left.length > 0:
            records.append({"origin_id": idx, "side": "left", "geometry": left, **base_props})
        if right and right.length > 0:
            records.append({"origin_id": idx, "side": "right", "geometry": right, **base_props})
    if not records:
        return gpd.GeoDataFrame(columns=["origin_id","side","geometry"], geometry="geometry", crs=gdf_roads.crs)
    gdf_sw = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf_roads.crs)
    # deterministic ordering: sort by origin_id then side (left before right)
    gdf_sw["origin_id_str"] = gdf_sw["origin_id"].astype(str)
    gdf_sw = gdf_sw.sort_values(["origin_id_str","side"]).drop(columns=["origin_id_str"])
    gdf_sw = gdf_sw.reset_index(drop=True)
    return gdf_sw
