# config.py
from pathlib import Path

# OVERPASS API configuration (Milano bounding box: south, west, north, east)
MILANO_BBOX = [45.386, 9.040, 45.535, 9.278]

OVERPASS_QUERIES = {
		"autograph": """
(
	way["highway"]["highway"!~"^(footway|path|steps|pedestrian|cycleway|corridor)$"]({bbox});
)
""".strip(),
		"crossings": """
(
	node["highway"="crossing"]({bbox});
	way["footway"="crossing"]({bbox});
	node["highway"="traffic_signals"]["crossing"~"^(yes|zebra|traffic_signals)$"]({bbox});
)
""".strip(),
		"pedestrian": """
(
	way["highway"~"^(footway|pedestrian|steps|path|living_street|residential)$"]({bbox});
	way["footway"="sidewalk"]({bbox});
	way["sidewalk"]({bbox});
	way["highway"="path"]["foot"!="no"]({bbox});
	way["highway"="cycleway"]["foot"~"yes|designated"]({bbox});
	way["highway"="service"]["access"~"^(yes|permissive|destination|customers|public)$"]({bbox});
	way["highway"="corridor"]({bbox});
	way["indoor"="corridor"]({bbox});
)
""".strip(),
		"amenities": """
(
	node["amenity"~"^(school|university|college|kindergarten)$"]({bbox});
	node["amenity"~"^(hospital|clinic|doctors|pharmacy)$"]({bbox});
	node["amenity"~"^(cafe|restaurant|bar|fast_food|pub)$"]({bbox});
	node["shop"]({bbox});
	node["tourism"~"^(museum|gallery|attraction)$"]({bbox});
	node["leisure"~"^(park|playground|pitch|sports_centre)$"]({bbox});
)
""".strip(),
}

# Buffers and geometry params (meters)
SIDEWALK_OFFSET_M = 1.5        # distance to offset centerline to create sidewalk lines
CROSSING_SEARCH_M = 30.0       # radius to search from crossing point to find sidewalks
SNAP_TOL_M = 1.0               # snapping tolerance for node coordinates (meters) - increased for GPS noise
MIN_COMPONENT_LENGTH_M = 5.0   # drop tiny components
PROXIMITY_BUFFER_M = 50.0      # buffer around major roads to check for pedestrian side roads

# Deterministic snapping precision: round coordinates to this many decimals (in meters)
SNAP_ROUND_DECIMALS = 3  # 0.001 m precision (millimeter) — deterministic

# Spatial tiling / temp
WORK_CRS_EPSG = 32632  # UTM Zone 32N for Bologna (better metric accuracy than 3857)

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
