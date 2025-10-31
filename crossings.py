# crossings.py
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from utils import geom_midpoint_on_line, snap_coord_tuple
from config import CROSSING_SEARCH_M, SNAP_TOL_M, SNAP_ROUND_DECIMALS

def insert_crossings(gdf_sidewalks, gdf_crossings, search_radius):
    """
    For each crossing point find the two nearest sidewalk segments (left/right sides)
    within search_radius. Prefer connectors between DIFFERENT origin_ids (different roads).
    Deterministic tie-break by sorted origin_id then side.
    Create connector LineStrings between nearest points on sidewalks.
    Returns connectors GeoDataFrame.
    """
    if gdf_crossings is None or gdf_crossings.empty:
        return gdf_sidewalks.iloc[0:0].copy()  # empty gdf with same crs

    sw = gdf_sidewalks.copy()
    sw = sw.set_geometry("geometry")
    pts = gdf_crossings.set_geometry("geometry").copy()

    connectors = []
    # build spatial index for sidewalks
    sw_sindex = sw.sindex

    for c_idx, c_row in pts.iterrows():
        point = c_row.geometry
        if point is None:
            continue
        # query bounding box
        buffer = point.buffer(search_radius)
        candidate_idx = list(sw_sindex.intersection(buffer.bounds))
        if not candidate_idx:
            continue
        candidates = sw.iloc[candidate_idx].copy()
        # compute nearest point on each candidate line and distance
        candidates["nearest_pt"] = candidates.geometry.apply(lambda g: nearest_points(point, g)[1])
        candidates["dist_to_cross"] = candidates["nearest_pt"].apply(lambda p: point.distance(p))
        # filter within radius
        candidates = candidates[candidates["dist_to_cross"] <= search_radius]
        if candidates.empty:
            continue
        # deterministic ordering: sort by origin_id then side then distance
        # ensure columns exist
        if "origin_id" in candidates.columns:
            candidates = candidates.sort_values(["origin_id","side","dist_to_cross"])
        else:
            candidates = candidates.sort_values(["dist_to_cross"])
        # pick two distinct sides (prefer left+right from DIFFERENT origins if possible)
        left_candidates = candidates[candidates["side"] == "left"]
        right_candidates = candidates[candidates["side"] == "right"]
        chosen = None
        
        if not left_candidates.empty and not right_candidates.empty:
            # Try to pick pair from DIFFERENT origins first (crossing between roads)
            left_origins = set(left_candidates["origin_id"])
            right_origins = set(right_candidates["origin_id"])
            diff_origin_pairs = [(l, r) for l in left_origins for r in right_origins if l != r]
            
            if diff_origin_pairs:
                # Pick closest pair from different roads (by combined distance)
                best_pair = None
                best_dist = float('inf')
                for oid_l, oid_r in diff_origin_pairs:
                    left_cands = left_candidates[left_candidates["origin_id"] == oid_l]
                    right_cands = right_candidates[right_candidates["origin_id"] == oid_r]
                    combined_dist = left_cands["dist_to_cross"].min() + right_cands["dist_to_cross"].min()
                    if combined_dist < best_dist:
                        best_dist = combined_dist
                        best_pair = (oid_l, oid_r)
                
                oid_l, oid_r = best_pair
                left = left_candidates[left_candidates["origin_id"] == oid_l].iloc[0]
                right = right_candidates[right_candidates["origin_id"] == oid_r].iloc[0]
                chosen = (left, right)
            else:
                # Fallback: same origin (crossing same road - less common but possible)
                common_origins = left_origins.intersection(right_origins)
                if common_origins:
                    oid = sorted(common_origins)[0]  # deterministic pick lowest origin_id
                    left = left_candidates[left_candidates["origin_id"] == oid].iloc[0]
                    right = right_candidates[right_candidates["origin_id"] == oid].iloc[0]
                    chosen = (left, right)
                else:
                    # Pick closest from each side regardless
                    left = left_candidates.iloc[0]
                    right = right_candidates.iloc[0]
                    chosen = (left, right)
        else:
            # if only one side present, attempt to pick two closest distinct sidewalk segments
            if len(candidates) >= 2:
                a = candidates.iloc[0]; b = candidates.iloc[1]
                chosen = (a, b)
            else:
                continue

        a, b = chosen
        pa = a["nearest_pt"]; pb = b["nearest_pt"]
        # build connector
        conn = LineString([pa, pb])
        # deterministic ID: use crossing index and snapped midpoint coords
        mid = conn.interpolate(0.5, normalized=True)
        sx, sy = round(mid.x, SNAP_ROUND_DECIMALS), round(mid.y, SNAP_ROUND_DECIMALS)
        connector_id = f"cross_{int(c_idx)}_{sx}_{sy}"
        connectors.append({
            "connector_id": connector_id,
            "from_origin": int(a["origin_id"]),
            "to_origin": int(b["origin_id"]),
            "geometry": conn,
            "crossing_index": int(c_idx)
        })
    if not connectors:
        return gdf_sidewalks.iloc[0:0].copy()
    gdf_conn = gpd.GeoDataFrame(connectors, geometry="geometry", crs=gdf_sidewalks.crs)
    return gdf_conn
