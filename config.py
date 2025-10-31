# config.py
from pathlib import Path

# Input CSVs (adjust paths if needed)
AUTOGRAPH_CSV = Path("autograph.csv")
PEDESTRIAN_CSV = Path("pedestrian.csv")   # for validation/optional
AMENITIES_CSV = Path("amenities.csv")     # optional
CROSSINGS_CSV = Path("crossings.csv")

# Buffers and geometry params (meters)
SIDEWALK_OFFSET_M = 1.5        # distance to offset centerline to create sidewalk lines
CROSSING_SEARCH_M = 30.0       # radius to search from crossing point to find sidewalks
SNAP_TOL_M = 1.0               # snapping tolerance for node coordinates (meters) - increased for GPS noise
MIN_COMPONENT_LENGTH_M = 5.0   # drop tiny components
PROXIMITY_BUFFER_M = 50.0      # buffer around major roads to check for pedestrian side roads

# Deterministic snapping precision: round coordinates to this many decimals (in meters)
SNAP_ROUND_DECIMALS = 3  # 0.001 m precision (millimeter) — deterministic

# Spatial tiling / temp
WORK_CRS_EPSG = 3857  # meters

# Exports
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)
OUT_NODES = OUT_DIR / "ped_nodes.geojson"
OUT_EDGES = OUT_DIR / "ped_edges.geojson"
OUT_GRAPHML = OUT_DIR / "ped_graph.graphml"

# Classification rule sets (prioritized)
NON_PEDESTRIAN_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
ALWAYS_PEDESTRIAN_HIGHWAYS = {"footway", "path", "pedestrian", "steps", "track"}
POSSIBLY_PEDESTRIAN = {"residential", "service", "unclassified", "living_street", "tertiary", "secondary"}
