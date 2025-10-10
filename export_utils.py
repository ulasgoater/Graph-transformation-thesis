"""Export Utilities Module

Purpose: Centralize all disk export logic (GeoJSON/CSV/JSON) for model outputs.
Separating these routines keeps core modeling code cleaner and simplifies
future maintenance (e.g., adding Parquet exports or cloud uploads).

Key design notes:
- We defensively strip extra geometry columns because GeoPandas writes only one.
- We always reproject to EPSG:4326 (WGS84) for web map compatibility.
- Functions are tolerant; they catch exceptions and print messages instead of
    halting the entire pipeline (export is usually a terminal step).
"""

import geopandas as gpd
from config import OUT_ALL_GEOJSON, OUT_PRED_GEOJSON


def export_predictions(gdf_result, out_all=OUT_ALL_GEOJSON, out_pred=OUT_PRED_GEOJSON):
    """Export prediction results to GeoJSON files.

    Writes two layers:
    1. Full set with prediction probabilities and labels.
    2. Filtered subset where model predicted sidewalk (pred_label_oof == 1).

    Parameters
    ----------
    gdf_result : GeoDataFrame
        Contains model outputs (pred_label_oof, pred_prob_oof) and geometry.
    out_all : str
        Destination for complete dataset (GeoJSON).
    out_pred : str
        Destination for positive predictions only (GeoJSON).
    """
    try:
        # Identify all geometry dtype columns
        geom_cols = [c for c in gdf_result.columns if hasattr(gdf_result[c], 'geom_type')]
        # Keep only the main geometry column
        extra_geom_cols = [c for c in geom_cols if c != gdf_result.geometry.name]
        if extra_geom_cols:
            gdf_export = gdf_result.drop(columns=extra_geom_cols)
        else:
            gdf_export = gdf_result

        gdf_export.to_crs(epsg=4326).to_file(out_all, driver="GeoJSON")

        predicted_segments = gdf_export[gdf_export["pred_label_oof"] == 1]
        predicted_segments.to_crs(epsg=4326).to_file(out_pred, driver="GeoJSON")
        print(f"Exported predictions: {out_all}, {out_pred}")
    except Exception as e:
        print("Export failed:", e)


def create_test_subset(gdf_auto, gdf_ped, gdf_amen, gdf_cross, subset_size_m=1000):
    """Create a square spatial subset centered on the first autograph segment.

    Useful for quick exploratory mapping or manual QA without loading the full
    extent. The subset is defined as a square of side length 2 * subset_size_m.

    Parameters
    ----------
    gdf_auto, gdf_ped, gdf_amen, gdf_cross : GeoDataFrame
        Source layers (must share same CRS in meters for bounding logic).
    subset_size_m : float
        Half-width of the square selection in meters.

    Returns
    -------
    (GeoDataFrame, GeoDataFrame, GeoDataFrame, GeoDataFrame)
        Filtered (auto, ped, amen, cross) layers.
    """
    if len(gdf_auto) == 0:
        raise RuntimeError("No autograph rows to test.")
    
    # Get center point from first autograph segment
    cent = gdf_auto.geometry.centroid.iloc[0]
    cx, cy = cent.x, cent.y
    half = subset_size_m
    
    # Define bounding box
    minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
    
    def within_bounds(gdf):
        """Return rows whose centroid falls inside the bounding box."""
        centroids = gdf.geometry.centroid
        mask = (
            centroids.x.between(minx, maxx) & 
            centroids.y.between(miny, maxy)
        )
        return gdf[mask].copy()
    
    # Apply filter to all datasets
    gdf_auto_subset = within_bounds(gdf_auto)
    gdf_ped_subset = within_bounds(gdf_ped)
    gdf_amen_subset = within_bounds(gdf_amen)
    gdf_cross_subset = within_bounds(gdf_cross)
    
    print("TEST subset sizes:", 
          len(gdf_auto_subset), len(gdf_ped_subset), 
          len(gdf_amen_subset), len(gdf_cross_subset))
    
    return gdf_auto_subset, gdf_ped_subset, gdf_amen_subset, gdf_cross_subset