"""Feature Engineering Module

This file transforms raw spatial data (roads, pedestrian ways, amenities, crossings)
into machine-learning ready tabular features AND creates the target label.

High-level steps:
1. Copy input road GeoDataFrame so we never mutate original data.
2. Compute helper geometries (centroids, buffers) used for spatial joins.
3. Create labels (legacy buffer-based AND refined distance-based labeling).
4. Derive feature groups: basic road attributes, crossings, amenities, derived metrics.
5. Return: enriched GeoDataFrame + lists of numeric and categorical feature column names.

Beginner note: A GeoDataFrame is like a regular pandas DataFrame but each row
also has a geometry (Point, LineString, etc.) and a coordinate reference system (CRS).
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    LABEL_BUFFER_M, CROSSING_BUFFER_M, AMENITY_BUFFER_SMALL_M, AMENITY_BUFFER_LARGE_M,
    DIST_LABEL_POS_THRESHOLD_M, DIST_LABEL_NEG_THRESHOLD_M,
    USE_ONLY_LINEAR_PED_FOR_LABEL, PRESERVE_LEGACY_LABEL, DROP_UNKNOWN_LABELS,
    ADAPTIVE_ADD_NEGATIVES, MIN_NEGATIVE_RATIO, MAX_ADAPTIVE_NEG_ADD
)
import warnings


def _with_left_index(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return a DataFrame with a guaranteed 'left_index' column after sjoin.

    Handles GeoPandas differences:
    - Some versions emit 'index_left' explicitly
    - Others use the left index as the joined index
    We normalize by resetting the index and adding a 'left_index' column.
    """
    if joined.empty:
        return joined
    j = joined.reset_index()
    if "index_left" in j.columns:
        j["left_index"] = j["index_left"]
    else:
        # After reset_index, the previous index becomes 'index'
        # which we treat as the left-side index
        if "index" not in j.columns:
            j = j.reset_index()
        j["left_index"] = j["index"]
    return j


def engineer_and_label(gdf_auto, gdf_ped, gdf_amen, gdf_cross,
                       label_buffer_m=LABEL_BUFFER_M,
                       crossing_buffer_m=CROSSING_BUFFER_M,
                       amen_small_m=AMENITY_BUFFER_SMALL_M,
                       amen_big_m=AMENITY_BUFFER_LARGE_M):
    """
    Engineer features and create labels for pedestrian infrastructure prediction.
    
    Args:
        gdf_auto: GeoDataFrame of road segments
        gdf_ped: GeoDataFrame of pedestrian infrastructure
        gdf_amen: GeoDataFrame of amenities
        gdf_cross: GeoDataFrame of crossings
        label_buffer_m: Buffer distance for labeling (meters)
        crossing_buffer_m: Buffer distance for crossing features (meters)
        amen_small_m: Small buffer for amenity features (meters)
        amen_big_m: Large buffer for amenity features (meters)
    
    Returns:
        tuple: (gdf_features, numeric_features, categorical_features)
    """
    gdf = gdf_auto.copy()
    
    # Create buffers and centroids
    gdf["centroid"] = gdf.geometry.centroid  # representative point to measure distances
    gdf["buf_label"] = gdf.geometry.buffer(label_buffer_m)
    gdf["buf_cross"] = gdf.geometry.buffer(crossing_buffer_m)
    gdf["buf_amen_small"] = gdf.geometry.buffer(amen_small_m)
    gdf["buf_amen_big"] = gdf.geometry.buffer(amen_big_m)

    # Legacy label (buffer-based) retained optionally
    label_meta = {}
    if PRESERVE_LEGACY_LABEL:
        gdf = _create_labels(gdf, gdf_ped, label_col="legacy_has_ped")

    # Distance-based labeling (supersedes legacy for model target)
    gdf, dist_meta = _apply_distance_based_label(gdf, gdf_ped)
    # If no negatives (all 1) and adaptive augmentation enabled, attempt to convert
    # farthest ambiguous (previously unknown) distances into negatives to ensure
    # at least minimal negative representation.
    if ADAPTIVE_ADD_NEGATIVES:
        neg_ratio = (gdf['has_ped'] == 0).mean() if 'has_ped' in gdf.columns else 0
        if neg_ratio == 0:
            # Look for rows that were previously unknown (we marked as -1 before dropping) only if we retained them
            # If unknowns were dropped, we cannot recover them here.
            if not DROP_UNKNOWN_LABELS and 'ped_line_dist_m' in gdf.columns:
                # Define candidates as those with largest distances not already positive
                cand = gdf[gdf['has_ped'] == -1].copy()
                if not cand.empty:
                    needed = int(max(1, MIN_NEGATIVE_RATIO * len(gdf)))
                    needed = min(needed, MAX_ADAPTIVE_NEG_ADD, len(cand))
                    cand_sorted = cand.sort_values('ped_line_dist_m', ascending=False).head(needed)
                    gdf.loc[cand_sorted.index, 'has_ped'] = 0
                    dist_meta['adaptive_negatives_added'] = int(len(cand_sorted))
                else:
                    dist_meta['adaptive_negatives_added'] = 0
            else:
                dist_meta['adaptive_negatives_added'] = 0
        else:
            dist_meta['adaptive_negatives_added'] = 0
    label_meta.update(dist_meta)
    
    # Engineer basic features
    gdf = _engineer_basic_features(gdf)
    
    # Engineer crossing features
    gdf = _engineer_crossing_features(gdf, gdf_cross)
    
    # Engineer amenity features
    gdf, amen_count_cols = _engineer_amenity_features(gdf, gdf_amen)
    
    # Engineer derived features
    gdf = _engineer_derived_features(gdf)
    
    # Clean up buffer columns
    for c in ["buf_label", "buf_cross", "buf_amen_small", "buf_amen_big"]:
        if c in gdf.columns:
            gdf.drop(columns=[c], inplace=True)
    
    # Define feature lists
    numeric_feats, categorical_feats = _define_feature_lists(gdf, amen_count_cols)
    
    return gdf, numeric_feats, categorical_feats, label_meta


def _create_labels(gdf, gdf_ped, label_col="has_ped"):
    """Create binary labels based on intersection with pedestrian infrastructure."""
    # Label: intersects pedestrian geometry
    # IMPORTANT: After renaming the geometry column we must explicitly set it as active,
    # otherwise GeoPandas loses track and CRS-based operations (sjoin) will fail.
    auto_buf = gdf[["buf_label"]].copy().rename(columns={"buf_label": "geometry"})
    auto_buf = gpd.GeoDataFrame(auto_buf, geometry="geometry", crs=gdf.crs)
    joined = gpd.sjoin(auto_buf, gdf_ped[["geometry"]], how="left", predicate="intersects")
    if not joined.empty:
        j = _with_left_index(joined)
        has_ped_idx = set(j["left_index"].unique())
    else:
        has_ped_idx = set()
    gdf[label_col] = gdf.index.to_series().apply(lambda i: 1 if i in has_ped_idx else 0).astype(int)
    # Early memory cleanup for buffer
    if "buf_label" in gdf.columns:
        gdf.drop(columns=["buf_label"], inplace=True)
    return gdf


def _apply_distance_based_label(gdf, gdf_ped):
    """Generate distance-based labels using line-only pedestrian geometries and tag overrides."""
    # Extract line geometries if configured
    if USE_ONLY_LINEAR_PED_FOR_LABEL:
        ped_lines = gdf_ped[gdf_ped.geometry.type.str.lower().str.contains("line")].copy()
        if ped_lines.empty:
            ped_lines = gdf_ped.copy()
    else:
        ped_lines = gdf_ped.copy()

    # Compute centroid distance to nearest pedestrian line
    try:
        nearest = gpd.sjoin_nearest(
            gdf.set_geometry("centroid"), ped_lines[["geometry"]], how="left", distance_col="ped_line_dist_m"
        )
        gdf["ped_line_dist_m"] = nearest["ped_line_dist_m"]
    except Exception as e:
        warnings.warn(f"sjoin_nearest failed for distance labeling: {e}. Setting distances NaN.")
        gdf["ped_line_dist_m"] = np.nan

    # Tag-based strong positive override
    tag_positive = np.zeros(len(gdf), dtype=bool)
    for tag in ["tags_sidewalk", "tags_sidewalk_left", "tags_sidewalk_right", "tags_sidewalk_both"]:
        if tag in gdf.columns:
            tag_positive |= gdf[tag].notna() & (gdf[tag].astype(str).str.lower().ne("no"))

    # Determine initial label categories
    dist = gdf["ped_line_dist_m"]
    pos_mask = (dist <= DIST_LABEL_POS_THRESHOLD_M) | tag_positive
    neg_mask = (~pos_mask) & (dist >= DIST_LABEL_NEG_THRESHOLD_M) & (~tag_positive)
    unknown_mask = ~(pos_mask | neg_mask)

    # Assign labels: 1 positive, 0 negative, -1 unknown
    gdf["has_ped"] = np.where(pos_mask, 1, np.where(neg_mask, 0, -1))

    meta = {
        "initial_count": int(len(gdf)),
        "unknown_count": int(unknown_mask.sum()),
        "positives_initial": int(pos_mask.sum()),
        "negatives_initial": int(neg_mask.sum()),
        "drop_unknowns": bool(DROP_UNKNOWN_LABELS)
    }
    if DROP_UNKNOWN_LABELS:
        before = len(gdf)
        gdf = gdf[gdf["has_ped"] != -1].copy()
        removed = before - len(gdf)
        meta["removed_unknown_count"] = int(removed)
        meta["final_count"] = int(len(gdf))
        meta["unknown_removed_fraction"] = float(removed / before) if before else 0.0
        print(f"Distance labeling: dropped {removed} ambiguous segments ({removed/max(before,1):.2%}).")
    else:
        meta["removed_unknown_count"] = 0
        meta["final_count"] = int(len(gdf))
        meta["unknown_removed_fraction"] = 0.0
        print(f"Distance labeling: positives={pos_mask.sum()} negatives={neg_mask.sum()} unknown={unknown_mask.sum()}")

    return gdf, meta


def _engineer_basic_features(gdf):
    """Engineer basic features from road attributes."""
    gdf["length_m"] = gdf.geometry.length
    gdf["highway"] = gdf.get("tags_highway", "unknown").fillna("unknown").astype(str)

    def parse_maxspeed(x):
        if pd.isna(x):
            return np.nan
        m = re.search(r"(\d+)", str(x))
        return float(m.group(1)) if m else np.nan

    gdf["maxspeed"] = gdf.get("tags_maxspeed").apply(parse_maxspeed) if "tags_maxspeed" in gdf.columns else np.nan
    gdf["num_lanes"] = pd.to_numeric(gdf.get("tags_lanes", np.nan), errors="coerce")
    gdf["is_oneway"] = gdf.get("tags_oneway", "no").astype(str).str.lower().eq("yes")
    gdf["lit"] = gdf.get("tags_lit", "no").astype(str).str.lower().eq("yes")
    gdf["surface"] = gdf.get("tags_surface", "unknown").fillna("unknown").astype(str)
    gdf["is_living_street"] = gdf.get("tags_living_street", "no").astype(str).str.lower().eq("yes")

    # Sidewalk tag features
    gdf["sidewalk_tag_left"] = gdf.get("tags_sidewalk_left").notna() if "tags_sidewalk_left" in gdf.columns else False
    gdf["sidewalk_tag_right"] = gdf.get("tags_sidewalk_right").notna() if "tags_sidewalk_right" in gdf.columns else False
    gdf["sidewalk_tag_both"] = (gdf.get("tags_sidewalk_both").notna() if "tags_sidewalk_both" in gdf.columns else False) | (gdf.get("tags_sidewalk_both", "") == "yes")
    gdf["sidewalk_tag_any"] = gdf[["sidewalk_tag_left", "sidewalk_tag_right", "sidewalk_tag_both"]].any(axis=1)

    return gdf


def _engineer_crossing_features(gdf, gdf_cross):
    """Engineer features related to crossings."""
    auto_cross_buf = gdf[["buf_cross"]].copy().rename(columns={"buf_cross": "geometry"})
    auto_cross_buf = gpd.GeoDataFrame(auto_cross_buf, geometry="geometry", crs=gdf.crs)
    joined_cross = gpd.sjoin(auto_cross_buf, gdf_cross[["geometry", "tags_crossing"]], how="left", predicate="intersects")
    if not joined_cross.empty:
        jc = _with_left_index(joined_cross)
        cross_counts = jc.groupby("left_index").size()
        gdf["cross_count_30m"] = cross_counts.reindex(gdf.index, fill_value=0)
    else:
        gdf["cross_count_30m"] = 0

    # Nearest crossing distance
    try:
        nearest_cross = gpd.sjoin_nearest(gdf.set_geometry("centroid"), gdf_cross[["geometry", "tags_crossing"]], how="left", distance_col="dist_cross_m")
        gdf["dist_cross_m"] = nearest_cross["dist_cross_m"]
    except Exception:
        gdf["dist_cross_m"] = np.nan

    # Signalized crossings
    if "tags_crossing" in gdf_cross.columns:
        gdf_cross_copy = gdf_cross.copy()
        gdf_cross_copy["crossing_simple"] = gdf_cross_copy["tags_crossing"].fillna("").astype(str).str.lower()
        sig = gdf_cross_copy[gdf_cross_copy["crossing_simple"].str.contains("traffic_signals|signal", na=False)]
        if not sig.empty:
            joined_sig = gpd.sjoin(auto_cross_buf, sig[["geometry"]], how="left", predicate="intersects")
            if not joined_sig.empty:
                js = _with_left_index(joined_sig)
                sig_counts = js.groupby("left_index").size()
                gdf["sig_cross_count_30m"] = sig_counts.reindex(gdf.index, fill_value=0)
            else:
                gdf["sig_cross_count_30m"] = 0
        else:
            gdf["sig_cross_count_30m"] = 0
    else:
        gdf["sig_cross_count_30m"] = 0

    return gdf


def _engineer_amenity_features(gdf, gdf_amen):
    """Engineer features related to amenities."""
    def map_amen_cat(r):
        a = str(r.get("tags_amenity", "")).lower()
        s = str(r.get("tags_shop", "")).lower()
        hc = str(r.get("tags_healthcare", "")).lower()
        school = str(r.get("tags_school", "")).lower()
        leisure = str(r.get("tags_leisure", "")).lower() if "tags_leisure" in r.index else ""
        
        if "school" in a or "school" in school:
            return "education"
        if a in ("bus_stop", "tram_stop", "subway_entrance") or "bus" in a:
            return "transport"
        if a in ("hospital", "clinic") or hc:
            return "healthcare"
        if s:
            return "retail"
        if a in ("park", "playground") or leisure:
            return "leisure"
        return "other"

    if "amen_cat" not in gdf_amen.columns:
        gdf_amen = gdf_amen.copy()
        gdf_amen["amen_cat"] = gdf_amen.apply(map_amen_cat, axis=1)

    amen_count_cols = []

    # 50m amenity counts
    auto_amen_50 = gdf[["buf_amen_small"]].copy().rename(columns={"buf_amen_small": "geometry"})
    auto_amen_50 = gpd.GeoDataFrame(auto_amen_50, geometry="geometry", crs=gdf.crs)
    joined_amen_50 = gpd.sjoin(auto_amen_50, gdf_amen[["geometry", "amen_cat"]], how="left", predicate="intersects")
    if not joined_amen_50.empty:
        j50 = _with_left_index(joined_amen_50)
        counts_50 = j50.groupby(["left_index", "amen_cat"]).size().unstack(fill_value=0)
        counts_50 = counts_50.reindex(gdf.index, fill_value=0)
        # Add distance-specific prefix to avoid column collision with 100m counts
        counts_50.columns = [f"amen50_{c}" for c in counts_50.columns]
        gdf = gdf.join(counts_50, how="left").fillna(0)
        gdf["amen_count_50m_all"] = counts_50.sum(axis=1).reindex(gdf.index, fill_value=0)
        amen_count_cols += counts_50.columns.tolist()
    else:
        gdf["amen_count_50m_all"] = 0

    # 100m amenity counts
    auto_amen_100 = gdf[["buf_amen_big"]].copy().rename(columns={"buf_amen_big": "geometry"})
    auto_amen_100 = gpd.GeoDataFrame(auto_amen_100, geometry="geometry", crs=gdf.crs)
    joined_amen_100 = gpd.sjoin(auto_amen_100, gdf_amen[["geometry", "amen_cat"]], how="left", predicate="intersects")
    if not joined_amen_100.empty:
        j100 = _with_left_index(joined_amen_100)
        counts_100 = j100.groupby(["left_index", "amen_cat"]).size().unstack(fill_value=0)
        counts_100 = counts_100.reindex(gdf.index, fill_value=0)
        counts_100.columns = [f"amen100_{c}" for c in counts_100.columns]
        gdf = gdf.join(counts_100, how="left").fillna(0)
        gdf["amen_count_100m_all"] = counts_100.sum(axis=1).reindex(gdf.index, fill_value=0)
        for c in counts_100.columns:
            if c not in amen_count_cols:
                amen_count_cols.append(c)
    else:
        gdf["amen_count_100m_all"] = 0

    # Nearest amenity distance
    try:
        nearest_amen = gpd.sjoin_nearest(gdf.set_geometry("centroid"), gdf_amen[["geometry", "amen_cat"]], how="left", distance_col="dist_amen_m")
        gdf["dist_amen_any_m"] = nearest_amen["dist_amen_m"]
    except Exception:
        gdf["dist_amen_any_m"] = np.nan
    # Amenity diversity (Shannon entropy) at 100m
    amen100_cols = [c for c in gdf.columns if c.startswith("amen100_")]
    if amen100_cols:
        counts = gdf[amen100_cols].fillna(0).astype(float)
        totals = counts.sum(axis=1)
        def entropy(row, total):
            if total <= 0:
                return 0.0
            p = row / total
            p = p[p > 0]
            return float(-(p * np.log(p)).sum())
        gdf["amenity_diversity_100m"] = [entropy(counts.iloc[i], totals.iloc[i]) for i in range(len(gdf))]
    else:
        gdf["amenity_diversity_100m"] = 0.0

    # Early memory cleanup for amenity buffers
    for c in ["buf_amen_small", "buf_amen_big"]:
        if c in gdf.columns:
            gdf.drop(columns=[c], inplace=True)
    return gdf, amen_count_cols


def _engineer_derived_features(gdf):
    """Engineer derived and heuristic features."""
    # Heuristic score
    def heuristic_score(r):
        sc = 0.0
        if any(x in str(r.get("highway", "")).lower() for x in ("residential", "service", "unclassified", "living_street")):
            sc += 0.3
        if pd.notna(r.get("maxspeed")) and r["maxspeed"] < 50:
            sc += 0.2
        if r.get("amen_count_50m_all", 0) > 0:
            sc += 0.2
        if r.get("cross_count_30m", 0) > 0 or (pd.notna(r.get("dist_cross_m")) and r["dist_cross_m"] <= 10):
            sc += 0.2
        if r.get("is_living_street", False):
            sc += 0.2
        return min(1.0, sc)

    gdf["heuristic_score"] = gdf.apply(heuristic_score, axis=1)

    # Curvature index: (length / chord) - 1
    def chord_len(geom):
        try:
            coords = list(geom.coords)
            if len(coords) < 2:
                return geom.length
            (x1,y1),(x2,y2) = coords[0], coords[-1]
            return ((x2-x1)**2 + (y2-y1)**2)**0.5
        except Exception:
            return np.nan
    chord_series = gdf.geometry.apply(chord_len).replace(0, np.nan)
    # Use safe division to avoid warnings and infs
    with np.errstate(divide='ignore', invalid='ignore'):
        sinuosity = np.divide(gdf.geometry.length, chord_series)
    gdf["curvature_index"] = (pd.Series(sinuosity, index=gdf.index) - 1).clip(lower=0).replace([np.inf, -np.inf], np.nan)

    # Boolean features
    gdf["has_amenity_50m"] = (gdf.get("amen_count_50m_all", 0) > 0).astype(int)
    gdf["has_amenity_100m"] = (gdf.get("amen_count_100m_all", 0) > 0).astype(int)
    gdf["has_crossing_30m"] = (gdf.get("cross_count_30m", 0) > 0).astype(int)
    gdf["has_signalized_crossing_30m"] = (gdf.get("sig_cross_count_30m", 0) > 0).astype(int)

    gdf["maxspeed_missing"] = gdf["maxspeed"].isna().astype(int)
    gdf["speed_lt_50"] = gdf["maxspeed"].lt(50).fillna(False).astype(int)

    # Distance-based features
    gdf["inv_dist_amen_any"] = gdf["dist_amen_any_m"].apply(lambda d: 1.0/(1.0 + d) if pd.notna(d) else 0.0)
    gdf["log_dist_amen_any"] = gdf["dist_amen_any_m"].apply(lambda d: np.log1p(d) if pd.notna(d) else np.nan)

    # Early memory cleanup for crossing buffer
    if "buf_cross" in gdf.columns:
        gdf.drop(columns=["buf_cross"], inplace=True)
    return gdf


def _define_feature_lists(gdf, amen_count_cols):
    """Define lists of numeric and categorical features."""
    # base_numeric: numeric features we attempt to use (only kept if column exists)
    base_numeric = [
        "length_m", "maxspeed", "num_lanes", "dist_cross_m", "dist_amen_any_m", 
        "inv_dist_amen_any", "log_dist_amen_any", "amen_count_50m_all", "amen_count_100m_all",
        "cross_count_30m", "sig_cross_count_30m", "heuristic_score", "ped_line_dist_m",
        "has_amenity_50m", "has_crossing_30m", "speed_lt_50", "maxspeed_missing",
        "curvature_index", "amenity_diversity_100m"
    ]
    
    # Include amenity category columns
    numeric_feats = [c for c in base_numeric if c in gdf.columns]
    for ac in amen_count_cols:
        if ac not in numeric_feats and ac in gdf.columns:
            numeric_feats.append(ac)

    categorical_feats = [
        c for c in ["highway", "surface", "is_oneway", "lit", "is_living_street", "sidewalk_tag_any"] 
        if c in gdf.columns
    ]

    return numeric_feats, categorical_feats