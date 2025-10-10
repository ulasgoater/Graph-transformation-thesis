"""Configuration parameters for the pedestrian sidewalk prediction pipeline.

This file centralizes ALL tunable knobs so that:
1. Experiments are reproducible (record these values in your thesis appendix).
2. Code elsewhere can stay simple and just import what it needs.

Beginner note: Changing values here changes how far we look for features, how we
label data, how we train the model, and how we export results. Treat this as the
"control panel" of the project.
"""

# ---------------------------------------------------------------------------
# INPUT DATA FILES (raw CSV exports from earlier OSM / extraction steps)
# ---------------------------------------------------------------------------
# If you rename or relocate files, update these names. Keeping them here avoids
# scattering hard-coded strings through the code.
AUTOGRAPH_CSV = "autograph.csv"
PEDESTRIAN_CSV = "pedestrian.csv"
AMENITIES_CSV = "amenities.csv"
CROSSINGS_CSV = "crossings.csv"

# ---------------------------------------------------------------------------
# SPATIAL BUFFER DISTANCES (all in meters)
# ---------------------------------------------------------------------------
# We create circles or expansions around each road segment to count or join
# nearby objects (amenities, crossings, etc.). Larger buffers capture broader
# context but can dilute local signal or increase computation.
LABEL_BUFFER_M = 5
CROSSING_BUFFER_M = 30
AMENITY_BUFFER_SMALL_M = 50
AMENITY_BUFFER_LARGE_M = 100

# ---------------------------------------------------------------------------
# PEDESTRIAN GRAPH BUILDING
# ---------------------------------------------------------------------------
# After predicting which road segments have sidewalks, we build a network graph.
# SNAP_TOL_M: how close two endpoints must be to be merged into one node.
# MIN_COMPONENT_LENGTH_M: discard tiny disconnected fragments below this length.
SNAP_TOL_M = 2.0
MIN_COMPONENT_LENGTH_M = 10.0

# ---------------------------------------------------------------------------
# SPATIAL CROSS-VALIDATION
# ---------------------------------------------------------------------------
# Instead of random folds (which can leak spatial info), we partition the map
# into tiles. Each tile is like a "spatial fold" to get a more honest estimate.
TILE_SIZE_M = 500
N_SPLITS_WANTED = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# CLASS BALANCE / SUBSETTING LOGIC
# ---------------------------------------------------------------------------
# We sometimes start with a small window and expand until we have at least a
# minimal minority class proportion, to avoid training on all positives or all
# negatives (which teaches the model nothing).
CLASS_BALANCE_MIN_POSITIVE_RATIO = 0.05  # Minimum positive ratio desired before considering interventions
# Some modules reference MIN_CLASS_RATIO; keep an alias for backward compatibility.
MIN_CLASS_RATIO = CLASS_BALANCE_MIN_POSITIVE_RATIO

# Label / dataset quality controls
# Minimum proportion of minority class required; otherwise auto-adjust or abort
# ---------------------------------------------------------------------------
# OPTIONAL FILTERS FOR PEDESTRIAN FEATURES
# ---------------------------------------------------------------------------
# Narrowing pedestrian geometries avoids labeling everything positive in dense
# areas. Adjust lists or set to [] to disable filtering.
# Maximum attempts to enlarge subset window to find negatives
MAX_SUBSET_EXPANSION_STEPS = 4
# Factor by which to grow subset each expansion
# (Added default; was previously undocumented in the file)
SUBSET_GROWTH_FACTOR = 1.5
# ---------------------------------------------------------------------------
# NEGATIVE SAMPLING (EMERGENCY FALLBACK)
# ---------------------------------------------------------------------------
# If, after labeling, every segment is positive (or every segment negative), we
# cannot train a classifier. As a contingency we synthesize negatives from
# segments far from pedestrian infrastructure. This should rarely be needed if
# labeling thresholds are well chosen.

# Optional pedestrian tag filters (set to empty list to disable)
PEDESTRIAN_ALLOWED_HIGHWAY = ["footway", "path", "pedestrian"]  # used to narrow pedestrian geometries
PEDESTRIAN_ALLOWED_FOOTWAY = ["sidewalk", "crossing"]  # if footway tag column exists

# ---------------------------------------------------------------------------
# DISTANCE-BASED LABELING (REFINED STRATEGY)
# ---------------------------------------------------------------------------
# We define:
#   Positive  if distance <= DIST_LABEL_POS_THRESHOLD_M (or strong sidewalk tag)
#   Negative  if distance >= DIST_LABEL_NEG_THRESHOLD_M (and no strong tag)
#   Unknown   otherwise (removed if DROP_UNKNOWN_LABELS=True)
# Distances are centroid-to-nearest pedestrian line.
NEGATIVE_SAMPLING_ENABLED = True
NEGATIVE_SAMPLING_TARGET_NEG_RATIO = 0.15  # aim for at least 15% negatives if we have 0
NEGATIVE_SAMPLING_DIST_FACTOR = 1.5  # candidate if ped_dist_m > LABEL_BUFFER_M * this factor
# Upper bound on number of synthetic negatives added (safeguard)
NEGATIVE_SAMPLING_MAX = 500
# Labeling behavior toggles

# Distance-based labeling thresholds (meters)
DIST_LABEL_POS_THRESHOLD_M = 2.0   # <= this distance => positive (if no strong tag overrides)
DIST_LABEL_NEG_THRESHOLD_M = 7.0   # >= this distance (and no positive tags) => negative
# Drop ambiguous middle-band labels (between POS and NEG thresholds)
DROP_UNKNOWN_LABELS = False

# ---------------------------------------------------------------------------
# ADAPTIVE NEGATIVE AUGMENTATION
# ---------------------------------------------------------------------------
# When no (or too few) negatives survive distance labeling (dense network areas),
# we can adaptively reclassify the farthest ambiguous segments as negatives to
# guarantee a minimum negative representation, stabilizing model training.
MIN_NEGATIVE_RATIO = 0.05         # Target minimum fraction of negatives after labeling
ADAPTIVE_ADD_NEGATIVES = True     # Enable adaptive conversion of farthest unknowns to negatives
MAX_ADAPTIVE_NEG_ADD = 1000       # Safety cap on number of adapted negatives
# ---------------------------------------------------------------------------
# EXPORT FILENAMES (analysis artifacts)
# ---------------------------------------------------------------------------
# Labeling behavior
USE_ONLY_LINEAR_PED_FOR_LABEL = True
PRESERVE_LEGACY_LABEL = True  # keep legacy buffer-based label for comparison as 'legacy_has_ped'
# CSV summarizing label counts (refined + legacy if preserved)
LABEL_SUMMARY_CSV = "label_summary.csv"
# Optional distance stats (future extension)

# Feature importance export
FEATURE_IMPORTANCES_CSV = "feature_importances.csv"
# Metadata tag stored in metrics so we know which method produced results

# Distance stats export
# ---------------------------------------------------------------------------
# MODEL EXPLANATION / BASELINES
# ---------------------------------------------------------------------------
# We compare the primary RandomForest to simpler baselines for scientific rigor.
EXPORT_DISTANCE_STATS = True

# Labeling strategy descriptor (for metrics metadata)
LABEL_STRATEGY = "distance_thresholds"  # options: distance_thresholds, legacy_buffer_only

# ---------------------------------------------------------------------------
# PIPELINE VERSIONING / REPRODUCIBILITY
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "1.0.0"
# RandomForest max features setting (string or int). Common: 'sqrt', 'log2', or None
MAX_FEATURES = "sqrt"

# Additional internal model controls (kept here for transparency)
# How many features a RandomForest considers at each split; 'sqrt' is a common
# default balancing bias vs variance.

# Minimum positive ratio we prefer to see before deciding data is adequate.
# Baseline model & explanation settings
ENABLE_BASELINE_LOGREG = True  # compute logistic regression baseline metrics
PERMUTATION_IMPORTANCE_N_REPEATS = 5  # repeats for permutation importance (if enabled)
EXPORT_PERMUTATION_IMPORTANCE = True
PERMUTATION_IMPORTANCES_CSV = "permutation_importances.csv"

# Output file paths
OUT_ALL_GEOJSON = "all_segments_with_predictions.geojson"
OUT_PRED_GEOJSON = "predicted_sidewalks.geojson"