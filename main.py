"""Main Pipeline Orchestration

This script stitches together all steps:
1. Load raw CSV data and convert to GeoDataFrames.
2. (Optional) Restrict to a smaller spatial subset while ensuring both classes exist.
3. Engineer features + create labels.
4. Train model with spatial cross-validation and gather evaluation metrics.
5. Export predictions and build a pedestrian network graph from predicted positives.
6. Write a metrics JSON (includes config hash) for full reproducibility.

Beginner note: Each stage prints progress so you can follow the flow. If a stage
fails, examine the error and the associated module (feature_engineering, model, etc.).
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from data_loader import load_gdfs
from feature_engineering import engineer_and_label
from model import train_and_evaluate
from graph_builder import build_pedestrian_graph
from export_utils import export_predictions, create_test_subset
from config import (
    SNAP_TOL_M, MIN_COMPONENT_LENGTH_M,
    MIN_CLASS_RATIO, MAX_SUBSET_EXPANSION_STEPS, SUBSET_GROWTH_FACTOR,
    NEGATIVE_SAMPLING_ENABLED, NEGATIVE_SAMPLING_TARGET_NEG_RATIO,
    NEGATIVE_SAMPLING_DIST_FACTOR, LABEL_BUFFER_M, NEGATIVE_SAMPLING_MAX,
    LABEL_SUMMARY_CSV, PRESERVE_LEGACY_LABEL, LABEL_STRATEGY,
    DIST_LABEL_POS_THRESHOLD_M, DIST_LABEL_NEG_THRESHOLD_M
)
from config import EXCLUDE_LABEL_LEAK_FEATURES
import json
import os


def main(run_test_subset=False, subset_size_m=1000, auto_expand_subset=True, export_metrics_path="metrics.json"):
    """
    Main pipeline for pedestrian sidewalk prediction and graph building.
    
    Args:
        run_test_subset: Whether to run on a small test subset of data
    """
    print("Starting pedestrian graph analysis pipeline...")
    
    # Load data
    print("Loading data...")
    gdf_auto, gdf_ped, gdf_amen, gdf_cross = load_gdfs()

    # Create test subset if requested (helps quicker iterations / debugging)
    if run_test_subset:
        print("Creating test subset...")
        current_size = subset_size_m
        expansion_step = 0
        while True:
            gdf_auto_sub, gdf_ped_sub, gdf_amen_sub, gdf_cross_sub = create_test_subset(
                gdf_auto, gdf_ped, gdf_amen, gdf_cross, subset_size_m=current_size
            )
            # Quick provisional balance check: we approximate class ratio cheaply
            # (not full labeling) to avoid repeated heavy feature computations.
            if len(gdf_ped_sub) == 0 or len(gdf_auto_sub) == 0:
                pos_ratio = 0.0
            else:
                # Rough: treat all as positive if pedestrian layer very dense; else sample
                sample_auto = gdf_auto_sub.sample(min(50, len(gdf_auto_sub)), random_state=0)
                # Use bounding box overlap as quick proxy (cheap) before full buffer
                ped_bounds_union = gdf_ped_sub.total_bounds  # (minx,miny,maxx,maxy)
                within_bbox = sample_auto.geometry.centroid.x.between(ped_bounds_union[0], ped_bounds_union[2]) & \
                              sample_auto.geometry.centroid.y.between(ped_bounds_union[1], ped_bounds_union[3])
                approx_pos = within_bbox.sum()
                pos_ratio = approx_pos / len(sample_auto) if len(sample_auto) else 0.0
            minority_ratio = min(pos_ratio, 1 - pos_ratio)
            if (minority_ratio >= MIN_CLASS_RATIO) or not auto_expand_subset:
                gdf_auto, gdf_ped, gdf_amen, gdf_cross = gdf_auto_sub, gdf_ped_sub, gdf_amen_sub, gdf_cross_sub
                print(f"Subset accepted at size {current_size} m (approx pos_ratio={pos_ratio:.2f}).")
                break
            expansion_step += 1
            if expansion_step > MAX_SUBSET_EXPANSION_STEPS:
                print(f"Reached max expansion steps; proceeding with last subset (pos_ratio≈{pos_ratio:.2f}).")
                gdf_auto, gdf_ped, gdf_amen, gdf_cross = gdf_auto_sub, gdf_ped_sub, gdf_amen_sub, gdf_cross_sub
                break
            current_size = int(current_size * SUBSET_GROWTH_FACTOR)
            print(f"Class imbalance (approx pos_ratio={pos_ratio:.2f}) — expanding subset to {current_size} m ...")

    # Engineer features and labels (core transformation step)
    print("Engineering features...")
    gdf_feat, numeric_feats, categorical_feats, label_meta = engineer_and_label(
        gdf_auto, gdf_ped, gdf_amen, gdf_cross
    )
    if EXCLUDE_LABEL_LEAK_FEATURES:
        dropped = label_meta.get("dropped_label_leak_features", [])
        if dropped:
            print(f"Excluded potential label-leak features: {dropped}")

    # Export label summary (after distance-based labeling inside feature engineering)
    try:
        label_counts = gdf_feat['has_ped'].value_counts(dropna=False).rename('count').reset_index().rename(columns={'index':'label'})
        if PRESERVE_LEGACY_LABEL and 'legacy_has_ped' in gdf_feat.columns:
            legacy_counts = gdf_feat['legacy_has_ped'].value_counts().rename('legacy_count')
            label_counts = label_counts.merge(legacy_counts, left_on='label', right_index=True, how='left')
        label_counts.to_csv(LABEL_SUMMARY_CSV, index=False)
        print(f"Label summary exported -> {LABEL_SUMMARY_CSV}")
    except Exception as e:
        print(f"Warning: could not export label summary: {e}")
    # Compute unknown removal fraction (if legacy comparison kept)
    unknown_fraction = label_meta.get("unknown_removed_fraction")

    # Ensure ped_dist_m exists (nearest pedestrian distance)
    if "ped_dist_m" not in gdf_feat.columns:
        try:
            nearest_ped = gpd.sjoin_nearest(
                gdf_feat.set_geometry("centroid"), 
                gdf_ped[["geometry"]], 
                how="left", 
                distance_col="ped_dist_m"
            )
            gdf_feat["ped_dist_m"] = nearest_ped["ped_dist_m"]
        except Exception:
            gdf_feat["ped_dist_m"] = np.nan
            print("ped_dist_m not computed due to sjoin_nearest error.")

    # Train and evaluate model
    print("Training model...")
    # Log class distribution pre-training
    pos_count = int((gdf_feat.get("has_ped", 0) == 1).sum())
    total_count = int(len(gdf_feat))
    neg_count = int((gdf_feat.get("has_ped", 0) == 0).sum())
    print(f"Label distribution: positives={pos_count} negatives={neg_count} (total={total_count})")
    if pos_count == 0 or neg_count == 0:
        print("WARNING: Only one class present before training after distance labeling.")
        if NEGATIVE_SAMPLING_ENABLED:
            print("Attempting synthetic negative sampling fallback...")
            try:
                # Prefer the refined distance to pedestrian line
                dist_series = gdf_feat['ped_line_dist_m'] if 'ped_line_dist_m' in gdf_feat.columns else gdf_feat.get('ped_dist_m')
                if dist_series is None or dist_series.isna().all():
                    raise ValueError('No distance series available for negative sampling.')
                # Primary criterion: beyond the negative threshold buffer scaled by factor
                # Use the max of configured negative threshold and a scaled buffer to pick far candidates
                scaled_threshold = max(DIST_LABEL_NEG_THRESHOLD_M, LABEL_BUFFER_M * NEGATIVE_SAMPLING_DIST_FACTOR)
                candidates = gdf_feat[dist_series > scaled_threshold].index.tolist()
                if not candidates:
                    # Fallback: pick farthest K segments overall if none pass threshold
                    k = int(min(NEGATIVE_SAMPLING_MAX, max(1, NEGATIVE_SAMPLING_TARGET_NEG_RATIO * len(gdf_feat))))
                    candidates = dist_series.sort_values(ascending=False).head(k).index.tolist()
                target_neg = int(min(NEGATIVE_SAMPLING_MAX, max(1, NEGATIVE_SAMPLING_TARGET_NEG_RATIO * len(gdf_feat))))
                rng = np.random.default_rng(0)
                chosen = candidates if len(candidates) <= target_neg else rng.choice(candidates, target_neg, replace=False)
                gdf_feat.loc[chosen, 'has_ped'] = 0
                pos_count = int(gdf_feat['has_ped'].sum())
                neg_count = len(gdf_feat) - pos_count
                print(f"Synthetic negatives added. New distribution pos={pos_count} neg={neg_count}")
            except Exception as e:
                print(f"Negative sampling fallback failed: {e}")
        else:
            print("Negative sampling disabled; model may be trivial.")
    gdf_result, preproc, clf, metrics = train_and_evaluate(
        gdf_feat, numeric_feats, categorical_feats
    )

    # Export predictions
    print("Exporting predictions...")
    export_predictions(gdf_result)

    # Build pedestrian graph
    print("Building pedestrian graph...")
    try:
        G_multi, nodes_gdf, edges_gdf = build_pedestrian_graph(
            gdf_result, 
            pred_label_col="pred_label_oof", 
            prob_col="pred_prob_oof", 
            snap_tol_m=SNAP_TOL_M, 
            min_component_length_m=MIN_COMPONENT_LENGTH_M, 
            export_prefix="ped_graph"
        )
        print(f"Pedestrian graph built: nodes={len(nodes_gdf)}, edges={len(edges_gdf)}")
    except Exception as e:
        print("Pedestrian graph build failed:", e)

    print("Pipeline completed successfully!")
    print("Final metrics:", metrics)

    # Export metrics JSON for reproducibility (thesis artifact)
    try:
        metrics_payload = {
            "metrics": metrics,
            "n_segments": int(len(gdf_result)),
            "n_predicted_positive": int((gdf_result.get("pred_label_oof", 0) == 1).sum()),
            "class_balance": float((gdf_result.get("has_ped", 0) == 1).mean()),
            "numeric_features": numeric_feats,
            "categorical_features": categorical_feats,
            "label_strategy": LABEL_STRATEGY,
            "distance_label_thresholds": {
                "positive_m": DIST_LABEL_POS_THRESHOLD_M,
                "negative_m": DIST_LABEL_NEG_THRESHOLD_M
            },
            "unknown_removed_fraction": unknown_fraction,
            "label_meta": label_meta,
            "feature_count": len(numeric_feats) + len(categorical_feats)
        }
        with open(export_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, default=str)
        print(f"Metrics written to {export_metrics_path}")
    except Exception as e:
        print("Failed to write metrics JSON:", e)
    
    return gdf_result, preproc, clf, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pedestrian graph analysis pipeline")
    parser.add_argument("--full", action="store_true", help="Run on full dataset (ignore subset)")
    parser.add_argument("--subset-size", type=int, default=1000, help="Initial subset half-size in meters")
    parser.add_argument("--no-expand", action="store_true", help="Disable automatic subset expansion")
    parser.add_argument("--metrics", type=str, default="metrics.json", help="Path to write metrics JSON")
    args = parser.parse_args()
    main(
        run_test_subset=not args.full,
        subset_size_m=args.subset_size,
        auto_expand_subset=not args.no_expand,
        export_metrics_path=args.metrics
    )