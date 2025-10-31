# classify.py
from config import NON_PEDESTRIAN_HIGHWAYS, ALWAYS_PEDESTRIAN_HIGHWAYS, POSSIBLY_PEDESTRIAN
import geopandas as gpd
import pandas as pd

def classify_autograph(gdf_auto, gdf_parks=None, proximity_buffer_m=50):
    """
    Add deterministic column 'is_pedestrian_road' to gdf_auto (bool).
    Rules (applied in order):
      1. if highway in NON_PEDESTRIAN_HIGHWAYS => False (major roads excluded)
      2. elif highway in ALWAYS_PEDESTRIAN_HIGHWAYS => True (footways, paths, etc.)
      3. elif segment is inside a park polygon (gdf_parks provided) => True
      4. elif highway in POSSIBLY_PEDESTRIAN and within proximity_buffer_m of NON_PEDESTRIAN_HIGHWAYS => True
         (side roads near major roads are pedestrian)
      5. if sidewalk tag explicitly exists => True (override)
      6. else => False
    
    The rules use existing tag columns if present, e.g. 'tags/highway' or 'tags_highway'.
    """
    gdf = gdf_auto.copy()
    # normalize possible column names
    col_candidates = [c for c in gdf.columns if c.lower().endswith("highway") or c.lower().endswith("/highway")]
    hw_col = col_candidates[0] if col_candidates else None

    def highway_value(row):
        if hw_col and pd.notna(row.get(hw_col)):
            return str(row.get(hw_col)).lower()
        # fallback: try 'highway' simple column
        if "highway" in row and pd.notna(row["highway"]):
            return str(row["highway"]).lower()
        return ""

    gdf["is_pedestrian_road"] = False

    # rule 1: non-pedestrian highways (major roads)
    mask_non_ped = gdf.apply(lambda r: highway_value(r) in NON_PEDESTRIAN_HIGHWAYS, axis=1)
    gdf.loc[mask_non_ped, "is_pedestrian_road"] = False

    # rule 2: always pedestrian
    mask_always = gdf.apply(lambda r: highway_value(r) in ALWAYS_PEDESTRIAN_HIGHWAYS, axis=1)
    gdf.loc[mask_always, "is_pedestrian_road"] = True

    # rule 3: inside park polygons (if provided)
    if gdf_parks is not None and not gdf_parks.empty:
        # ensure same CRS
        gdfp = gdf_parks.to_crs(gdf.crs)
        joined = gpd.sjoin(gdf.set_geometry("geometry"), gdfp[["geometry"]], how="left", predicate="within")
        inside_idx = joined[~joined["index_right"].isna()].index.unique()
        gdf.loc[inside_idx, "is_pedestrian_road"] = True

    # rule 4: possibly pedestrian highways -> check proximity to major roads
    mask_poss = gdf.apply(lambda r: highway_value(r) in POSSIBLY_PEDESTRIAN, axis=1)
    possibly_ped_candidates = gdf[mask_poss & ~mask_always & ~gdf["is_pedestrian_road"]].copy()
    
    if not possibly_ped_candidates.empty and mask_non_ped.any():
        # Get major roads for proximity check
        major_roads = gdf[mask_non_ped].copy()
        if not major_roads.empty:
            print(f"  Checking {len(possibly_ped_candidates)} possibly-pedestrian roads against {len(major_roads)} major roads...")
            # Buffer major roads
            major_buffered = major_roads.geometry.buffer(proximity_buffer_m)
            # Check which possibly-pedestrian roads intersect buffered major roads
            for idx in possibly_ped_candidates.index:
                geom = possibly_ped_candidates.loc[idx, 'geometry']
                if any(geom.intersects(buff) for buff in major_buffered):
                    gdf.loc[idx, "is_pedestrian_road"] = True

    # rule 5: final override - if sidewalk tag explicitly exists (left/right/both) mark pedestrian
    sidewalk_cols = [c for c in gdf.columns if "sidewalk" in c.lower()]
    if sidewalk_cols:
        sw_mask = gdf[sidewalk_cols].notna().any(axis=1)
        gdf.loc[sw_mask, "is_pedestrian_road"] = True

    return gdf
