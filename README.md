# Pedestrian Graph Analysis Pipeline

This project implements a machine learning pipeline for predicting pedestrian infrastructure (sidewalks) from OpenStreetMap data and building pedestrian network graphs.

## Project Structure

The original monolithic script has been refactored into modular components following the single responsibility principle:

### Core Modules

- **`config.py`** - Configuration parameters and constants
- **`utils.py`** - Utility functions for data processing
- **`data_loader.py`** - CSV data loading and preprocessing
- **`feature_engineering.py`** - Feature engineering and labeling
- **`model.py`** - Machine learning model training and evaluation
- **`graph_builder.py`** - Pedestrian network graph construction
- **`export_utils.py`** - Data export and visualization utilities
- **`main.py`** - Main pipeline orchestration

### Data Files

The pipeline expects the following CSV files:
- `autograph.csv` - Road network data from OpenStreetMap
- `pedestrian.csv` - Pedestrian infrastructure data
- `amenities.csv` - Points of interest and amenities
- `crossings.csv` - Pedestrian crossing data

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required CSV data files in the project directory.

## Usage

### Basic Usage

    * Now supports CLI flags for reproducibility: `--full`, `--subset-size`, `--no-expand`, `--metrics`.
Run the complete pipeline with a test subset:

```python
from main import main

# Run with test subset (default)
results = main(run_test_subset=True)

# Run with full dataset
- **Class balance controls**: `MIN_CLASS_RATIO`, subset auto-expansion parameters
- **Pedestrian filtering**: `PEDESTRIAN_ALLOWED_HIGHWAY`, `PEDESTRIAN_ALLOWED_FOOTWAY` to avoid label saturation
results = main(run_test_subset=False)
```

### Individual Module Usage

You can also use individual modules for specific tasks:

```python
# Load data
from data_loader import load_gdfs
gdf_auto, gdf_ped, gdf_amen, gdf_cross = load_gdfs()

# Engineer features
from feature_engineering import engineer_and_label
gdf_feat, numeric_feats, categorical_feats = engineer_and_label(
    gdf_auto, gdf_ped, gdf_amen, gdf_cross

### Command Line Interface

Run with automatic subset (adaptive enlargement to reach class balance threshold):

```powershell
python main.py --subset-size 1000
```

Run full dataset without subsetting:

```powershell
python main.py --full
```

Disable automatic subset expansion:

```powershell
python main.py --subset-size 1500 --no-expand
```

Write metrics to a custom path:

```powershell
python main.py --full --metrics run1_metrics.json
```

Metrics JSON includes: model metrics, segment counts, class balance, and feature lists for thesis reproducibility.
)

# Train model
from model import train_and_evaluate
gdf_result, preproc, clf, metrics = train_and_evaluate(
    gdf_feat, numeric_feats, categorical_feats
)

# Build graph
from graph_builder import build_pedestrian_graph
G, nodes_gdf, edges_gdf = build_pedestrian_graph(gdf_result)
```

## Configuration

Key parameters can be adjusted in `config.py`:

- **Buffer distances**: Control feature engineering spatial joins
- **Graph parameters**: Snapping tolerance and minimum component length
- **Model parameters**: Cross-validation splits and random seed
- **File paths**: Input and output file locations

## Output Files

The pipeline generates several output files:

- `all_segments_with_predictions.geojson` - All road segments with predictions
- `predicted_sidewalks.geojson` - Only predicted sidewalk segments
- `ped_graph_nodes.geojson` - Network nodes
- `ped_graph_edges.geojson` - Network edges  
- `ped_graph.graphml` - Complete network graph
- `label_summary.csv` - Counts of refined labels (and legacy counts if preserved)
- `feature_importances.csv` - Ranked model feature importances after full-data retrain
- `metrics.json` (or user-specified) - Reproducibility bundle (metrics + feature lists + class balance + label strategy)
- Includes Brier score (probabilistic calibration) once model has two classes
- `distance_stats.json` (optional future extension) - Summary of distance distributions

## Features

### Machine Learning & Labeling Enhancements
- Spatial cross-validation (tile grouping) to reduce spatial leakage
- Distance + tag hybrid labeling:
    - Positive if centroid within `DIST_LABEL_POS_THRESHOLD_M` of a pedestrian line OR strong sidewalk tag.
    - Negative if beyond `DIST_LABEL_NEG_THRESHOLD_M` with no positive tags.
    - Ambiguous middle band removed when `DROP_UNKNOWN_LABELS=True` (reported in `label_summary.csv`).
- Legacy buffer label optionally preserved (`PRESERVE_LEGACY_LABEL=True`) for method comparison.
- Synthetic negative fallback only if refined labeling still produces a single class.
- Feature importances exported (`feature_importances.csv`).
- Metrics JSON enriched with labeling strategy metadata for thesis reproducibility.

### Graph Construction
- Automatic node snapping for network connectivity
- Component filtering by minimum length
- Multiple export formats (GeoJSON, GraphML, CSV)
- Handles MultiLineString geometries

### Spatial Features
- Distance to nearest amenities and crossings
- Distance to nearest pedestrian line (`ped_line_dist_m`) powering refined labeling
- Amenity density at 50m & 100m (prefixed columns: `amen50_`, `amen100_`)
- Road characteristics (speed, lanes, surface, living street, lighting)
- Pedestrian OSM tags & derived heuristic score

## Architecture Benefits

The modular design provides:

1. **Single Responsibility**: Each module has a clear, focused purpose
2. **Maintainability**: Easier to debug and modify individual components
3. **Reusability**: Modules can be used independently or in other projects
4. **Testability**: Individual functions can be unit tested
5. **Scalability**: Easy to add new features or modify existing ones
6. **Transparency**: Dual labeling (legacy vs refined) + exported summaries support auditability and thesis validation

## Requirements

- Python 3.7+
- pandas >= 1.3.0
6. **Reproducibility**: Deterministic randomness (`RANDOM_STATE`) + metrics JSON + explicit feature lists
- geopandas >= 0.10.0
- shapely >= 1.8.0
- networkx >= 2.6.0
- scikit-learn >= 1.0.0
- numpy >= 1.20.0
- rtree >= 0.9.0 (recommended for spatial operations)

## Formal Label Definition

Let a candidate road segment i have centroid c_i and distance d_i to the nearest
pedestrian infrastructure line (e.g. footway, path) measured in meters.

Let T_i be a boolean that is True if segment i carries an explicit strong
sidewalk/pedestrian tag (e.g. highway=footway, sidewalk in {both,left,right},
or dedicated pedestrian way) and False otherwise.

Given thresholds P = DIST_LABEL_POS_THRESHOLD_M and N = DIST_LABEL_NEG_THRESHOLD_M
with P < N, define the refined label L_i as:

    If (d_i <= P) OR T_i is True:         L_i = 1 (positive)
    Else if (d_i >= N) AND T_i is False:  L_i = 0 (negative)
    Else:                                 L_i = UNKNOWN

If DROP_UNKNOWN_LABELS=True the UNKNOWN class is removed prior to model
training; otherwise it may be preserved for diagnostic reporting (but is
excluded from supervised fitting). If PRESERVE_LEGACY_LABEL=True an additional
column legacy_label stores the original buffer/overlay-based label for
comparative analysis (enabling ablation in the thesis).

Synthetic negatives are only generated if, after removal (or retention) of
UNKNOWN, the label set collapses to a single class—ensuring the model can fit.

## Glossary

| Term | Meaning |
|------|---------|
| Autograph segment | A candidate road segment from OSM under evaluation for sidewalk presence |
| Pedestrian line | Existing mapped pedestrian facility (footway, path, sidewalk-tagged way) used as ground-truth anchor |
| Centroid distance d_i | Planar distance (meters) from segment centroid to nearest pedestrian line |
| Positive threshold P | Upper distance bound for automatic positives (DIST_LABEL_POS_THRESHOLD_M) |
| Negative threshold N | Lower distance bound beyond which (without tags) segment is labeled negative (DIST_LABEL_NEG_THRESHOLD_M) |
| Ambiguous band | Distance interval (P, N) where neither proximity nor remoteness is decisive; removed if DROP_UNKNOWN_LABELS=True |
| Strong tag T_i | Boolean indicating explicit pedestrian/sidewalk tag presence overriding distance logic in favor of positive |
| Refined label L_i | Final binary/unknown label after distance + tag hybrid logic |
| Legacy label | Original simplistic buffer-based label retained when PRESERVE_LEGACY_LABEL=True |
| Spatial CV | Cross-validation that groups geographically proximate segments to reduce spatial leakage |
| OOF prediction | Out-of-fold prediction; probability/label produced for a sample by a model that did not train on that sample |
| Permutation importance | Feature importance computed by measuring performance drop after random shuffling of a feature |
| Brier score | Mean squared error between predicted probability and actual label (lower is better) |
| Sinuosity / curvature_index | Ratio of line length to straight-line distance between endpoints; higher implies more curved |
| Amenity diversity | Shannon entropy of amenity category counts within a buffer (e.g. 100m) |
| Snapping tolerance | Distance (meters) within which distinct endpoints are merged into one graph node |
| Minimum component length | Threshold for total edge length required for a connected subgraph to be retained |
| Config hash | MD5 digest summarizing active configuration constants for reproducibility |
| Heuristic score | Composite engineered score aggregating multiple pedestrian-friendly indicators |

## Suggested Thesis Ablations

To strengthen methodological rigor, consider reporting:
- Distance-only baseline vs full feature set (already implemented).
- Legacy vs refined labels (accuracy/F1 deltas).
- Effect of removing ambiguous band (performance & class balance changes).
- Feature removal trials (e.g., without amenity diversity) to quantify marginal contribution.

## Reproducibility Checklist

1. Record config hash and PIPELINE_VERSION in metrics.json.
2. Archive feature_importances.csv and permutation importances when produced.
3. Store label_meta (class counts, thresholds, unknown fraction).
4. Log random seed and scikit-learn version (consider adding to metrics if not already).
5. Preserve raw input CSV snapshots or their hashes for audit trail.

## Troubleshooting

| Issue | Likely Cause | Remedy |
|-------|--------------|--------|
| All segments labeled positive | Thresholds too large or tags too permissive | Lower DIST_LABEL_POS_THRESHOLD_M or restrict strong tag list |
| No negatives after unknown removal | Narrow gap between P and N | Increase N or decrease P to widen ambiguous band |
| Export fails with multiple geometry columns | Intermediate joins added extra geometry | Use export utilities (they drop extras) |
| Graph too fragmented | snap_tol_m too small | Increase SNAP_TOL_M |
| Memory spikes during spatial joins | Very large buffers | Reduce buffer distances or pre-filter candidate layers |
