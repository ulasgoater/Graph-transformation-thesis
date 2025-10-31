Pedestrian Graph Transformation Pipeline - Deterministic
========================================================

## Overview
This pipeline transforms automotive road networks (autograph) into pedestrian-accessible graphs using **deterministic, rule-based methods**. No machine learning or probabilistic methods are used.

## Quick Start

1. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare input data**:
   - `autograph.csv` - Road network with OSM tags
   - `crossings.csv` - Pedestrian crossing points
   - `amenities.csv` (optional) - Points of interest

3. **Run pipeline**:
   ```bash
   python run_pipeline.py
   ```

4. **Outputs** (in `output/` directory):
   - `ped_nodes.geojson` - Graph nodes
   - `ped_edges.geojson` - Graph edges
   - `ped_graph.graphml` - NetworkX graph format

## Pipeline Stages

### 1. Road Classification (`classify.py`)
**Deterministic rules applied in order:**

1. **NON_PEDESTRIAN_HIGHWAYS** → Excluded
   - `motorway`, `motorway_link`, `trunk`, `trunk_link`, `primary`, `primary_link`
   
2. **ALWAYS_PEDESTRIAN_HIGHWAYS** → Included
   - `footway`, `path`, `pedestrian`, `steps`, `track`
   
3. **Inside park polygons** → Included
   - If `gdf_parks` provided and road geometry is within park boundary
   
4. **POSSIBLY_PEDESTRIAN + Proximity check** → Conditionally included
   - Road types: `residential`, `service`, `unclassified`, `living_street`, `tertiary`, `secondary`
   - **Only included if within 50m of a major road** (NON_PEDESTRIAN_HIGHWAYS)
   - Rationale: Side roads near major roads are pedestrian-accessible
   
5. **Sidewalk tag override** → Included
   - Any road with `sidewalk:left`, `sidewalk:right`, or `sidewalk:both` tags

**Configuration**: `PROXIMITY_BUFFER_M = 50.0` in `config.py`

### 2. Sidewalk Generation (`sidewalks.py`)
For each pedestrian road:
- **Normalize line direction**: West-to-east (or south-to-north if same longitude) for consistent left/right orientation
- **Create parallel offsets**: 
  - Left sidewalk: 1.5m offset to left of normalized direction
  - Right sidewalk: 1.5m offset to right
- **Handle MultiLineString**: Select longest component deterministically
- **Ordering**: Sort by `origin_id` then `side` ('left' before 'right')

**Configuration**: `SIDEWALK_OFFSET_M = 1.5` in `config.py`

### 3. Crossing Connectors (`crossings.py`)
For each crossing point:
- **Search radius**: 30m to find nearby sidewalk segments
- **Selection logic**:
  1. Prefer connectors between **different roads** (different `origin_id`)
  2. Pick closest left+right sidewalk pair by combined distance
  3. Fallback: Connect same road if no intersection found
- **Create LineString**: From nearest point on one sidewalk to nearest point on other

**Configuration**: `CROSSING_SEARCH_M = 30.0` in `config.py`

### 4. Graph Construction (`graph.py`)
**Node creation:**
- Extract endpoints from all sidewalk segments and connectors
- **Snap coordinates**: Round to multiples of `SNAP_TOL_M` (1.0m tolerance)
- **Precision**: Round to 3 decimals (1mm precision) for determinism
- **Unique node IDs**: `n_{index}_{x}_{y}`

**Edge creation:**
- Split sidewalks at:
  - Connector endpoints
  - **Sidewalk-sidewalk intersections** (T-junctions, crossroads)
- Each split segment becomes an edge
- Edge attributes: `edge_id`, `u` (from node), `v` (to node), `length_m`, `geometry`

**Component filtering:**
- Remove disconnected components < 5m total length
- Keeps only substantial connected networks

**Ordering:** Deterministic sort by `node_id` and `edge_id`

## Determinism Guarantees

✅ **Same input → Same output** guaranteed by:

1. **Sorted data processing**: All rows processed in sorted order (`origin_id`, `side`, `connector_id`)
2. **Fixed precision**: Coordinates rounded to 3 decimals (1mm) in meters
3. **No randomness**: No random sampling, shuffling, or probabilistic methods
4. **Stable geometry ops**: Direction normalization ensures consistent left/right
5. **Deterministic tie-breaking**: Lowest `origin_id` selected when multiple options exist

## Configuration Parameters

All constants in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIDEWALK_OFFSET_M` | 1.5 | Sidewalk distance from road centerline |
| `CROSSING_SEARCH_M` | 30.0 | Search radius for nearby sidewalks at crossings |
| `SNAP_TOL_M` | 1.0 | Node coordinate snapping tolerance (increased for GPS noise) |
| `MIN_COMPONENT_LENGTH_M` | 5.0 | Minimum length to keep connected components |
| `PROXIMITY_BUFFER_M` | 50.0 | Buffer around major roads for side road detection |
| `SNAP_ROUND_DECIMALS` | 3 | Coordinate precision (millimeters) |
| `WORK_CRS_EPSG` | 3857 | Working coordinate system (Web Mercator) |

## Validation

### Test Determinism
Run the pipeline twice and verify identical outputs:

```python
import hashlib
from run_pipeline import main

def hash_outputs():
    with open('output/ped_nodes.geojson', 'rb') as f:
        nodes_hash = hashlib.md5(f.read()).hexdigest()
    with open('output/ped_edges.geojson', 'rb') as f:
        edges_hash = hashlib.md5(f.read()).hexdigest()
    return nodes_hash, edges_hash

main()
hash1 = hash_outputs()
main()
hash2 = hash_outputs()
assert hash1 == hash2, "Not deterministic!"
print("✓ Determinism verified")
```

### Graph Quality Metrics
Recommended metrics for thesis validation:
- Number of nodes and edges
- Connected components count
- Average node degree (connectivity)
- Total network length (km)
- Crossing density (connectors per km²)
- Coverage compared to reference `pedestrian.csv`

## Architecture

```
autograph.csv ─┬─> classify.py ──> sidewalks.py ──┬─> graph.py ─> outputs
               │                                   │
crossings.csv ─┴──────────────────> crossings.py ─┘
```

**Files:**
- `run_pipeline.py` - Main orchestration
- `classify.py` - Road classification rules
- `sidewalks.py` - Parallel offset generation
- `crossings.py` - Connector creation
- `graph.py` - Node/edge graph construction with splitting
- `io_utils.py` - CSV/GeoJSON loading
- `utils.py` - Geometry utilities
- `config.py` - Configuration constants

## Notes

- **Test subset mode**: By default runs on 1km² area (set `run_test_subset=False` for full dataset)
- **Missing crossings**: Pipeline continues with warning if `crossings.csv` is empty
- **CRS handling**: All inputs converted to EPSG:3857 (meters), outputs in EPSG:4326 (lat/lon)
- **Error handling**: Geometry failures skip individual segments with logging

## Requirements

See `requirements.txt`:
- pandas
- geopandas >= 0.10
- shapely
- networkx
- numpy
- rtree (spatial indexing)
