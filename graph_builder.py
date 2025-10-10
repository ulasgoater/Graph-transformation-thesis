"""Graph Builder

Purpose: Turn predicted positive sidewalk segments into a network (nodes + edges)
so we can analyze connectivity, component sizes, and export a usable pedestrian
graph for downstream analysis or routing.

High-level steps:
1. Filter only segments predicted as sidewalk (binary label == 1).
2. Normalize geometry (pick longest part if MultiLineString).
3. Extract endpoints of each line.
4. Snap endpoints to a grid (approximate merging of very close nodes) to reduce noise.
5. Build a NetworkX MultiGraph with probability + length attributes.
6. Optionally remove very small disconnected components.
7. Export nodes, edges (GeoJSON) and aggregated GraphML.

Beginner note: A graph is a set of nodes (points) and edges (connections). We
use snapping so two endpoints a couple of meters apart become a single logical
intersection node.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString

from config import SNAP_TOL_M, MIN_COMPONENT_LENGTH_M


def build_pedestrian_graph(gdf_result, pred_label_col="pred_label_oof", prob_col="pred_prob_oof",
                           snap_tol_m=SNAP_TOL_M, min_component_length_m=MIN_COMPONENT_LENGTH_M,
                           export_prefix="ped_graph"):
    """
    Build a pedestrian network graph from predicted sidewalk segments.
    
    Args:
        gdf_result: GeoDataFrame with prediction results
        pred_label_col: Column name for binary predictions
        prob_col: Column name for prediction probabilities
        snap_tol_m: Snapping tolerance in meters
        min_component_length_m: Minimum component length to keep
        export_prefix: Prefix for output files
    
    Returns:
        tuple: (networkx_graph, nodes_geodataframe, edges_geodataframe)
    """
    # Filter predicted segments
    gdf_pred = gdf_result[gdf_result[pred_label_col] == 1].copy()
    if gdf_pred.empty:
        raise ValueError("No predicted pedestrian segments found (pred_label_oof == 1).")

    # Ensure LineString geometries
    gdf_pred = _ensure_linestrings(gdf_pred)  # sanitize geometries
    
    # Extract endpoints
    gdf_pred = _extract_endpoints(gdf_pred)
    
    # Create node mapping with snapping
    node_map = _create_node_mapping(gdf_pred, snap_tol_m)  # maps snap key -> snapped coordinate
    
    # Build NetworkX graph
    G = _build_networkx_graph(gdf_pred, node_map, snap_tol_m, prob_col)  # constructs MultiGraph
    
    # Create GeoDataFrames for nodes and edges
    nodes_gdf, edges_gdf = _create_geodataframes(G, gdf_pred, node_map, snap_tol_m, prob_col)
    
    # Filter by component length
    if min_component_length_m > 0:
        nodes_gdf, edges_gdf = _filter_by_component_length(
            G, nodes_gdf, edges_gdf, min_component_length_m
        )
    
    # Export files
    _export_graph_files(nodes_gdf, edges_gdf, export_prefix)
    
    return G, nodes_gdf, edges_gdf


def _ensure_linestrings(gdf_pred):
    """Ensure all geometries are LineStrings (choose longest if MultiLineString)."""
    def ensure_linestring(geom):
        if geom is None:
            return None
        if isinstance(geom, LineString):
            return geom
        if isinstance(geom, MultiLineString):
            parts = list(geom)
            lengths = [p.length for p in parts]
            return parts[int(np.argmax(lengths))]
        return None

    gdf_pred["geometry"] = gdf_pred.geometry.apply(ensure_linestring)
    gdf_pred = gdf_pred[gdf_pred.geometry.notna()].copy()
    return gdf_pred


def _extract_endpoints(gdf_pred):
    """Extract (first, last) coordinate pair for each line to form edge endpoints."""
    def endpoints_coords(ls):
        coords = list(ls.coords)
        return coords[0], coords[-1]

    gdf_pred["end1"], gdf_pred["end2"] = zip(*gdf_pred.geometry.apply(endpoints_coords))
    return gdf_pred


def _create_node_mapping(gdf_pred, snap_tol_m):
    """Create node mapping with coordinate snapping (grid-based clustering).

    This simple approach rounds coordinates to a grid multiple of the tolerance.
    Pros: Fast and deterministic. Cons: Can merge nodes across a diagonal within tol.
    """
    def snap_key(coord, tol):
        ix = int(round(coord[0] / tol))
        iy = int(round(coord[1] / tol))
        return (ix, iy)

    node_map = {}
    for idx, row in gdf_pred.iterrows():
        for pt in (row["end1"], row["end2"]):
            key = snap_key(pt, snap_tol_m)
            if key not in node_map:
                node_map[key] = (key[0] * snap_tol_m, key[1] * snap_tol_m)
    
    return node_map


def _build_networkx_graph(gdf_pred, node_map, snap_tol_m, prob_col):
    """Build NetworkX MultiGraph from predicted segments with length & probability attributes."""
    def snap_key(coord, tol):
        ix = int(round(coord[0] / tol))
        iy = int(round(coord[1] / tol))
        return (ix, iy)

    G = nx.MultiGraph()
    
    # Add nodes
    for key, (x, y) in node_map.items():
        G.add_node(key, x=float(x), y=float(y))

    # Add edges
    for idx, row in gdf_pred.iterrows():
        u = snap_key(row["end1"], snap_tol_m)
        v = snap_key(row["end2"], snap_tol_m)
        if u == v:
            continue
        
        geom = row.geometry
        length = float(geom.length)
        edge_id = str(idx)
        prob = float(row.get(prob_col, np.nan)) if pd.notna(row.get(prob_col, np.nan)) else np.nan
        
        G.add_edge(u, v, edge_id=edge_id, length_m=length, pred_prob=prob)
    
    return G


def _create_geodataframes(G, gdf_pred, node_map, snap_tol_m, prob_col):
    """Create GeoDataFrames for nodes (Point) and edges (LineString)."""
    def snap_key(coord, tol):
        ix = int(round(coord[0] / tol))
        iy = int(round(coord[1] / tol))
        return (ix, iy)

    # Create edges GeoDataFrame
    edge_records = []
    for idx, row in gdf_pred.iterrows():
        u = snap_key(row["end1"], snap_tol_m)
        v = snap_key(row["end2"], snap_tol_m)
        if u == v:
            continue
        
        geom = row.geometry
        length = float(geom.length)
        edge_id = str(idx)
        prob = float(row.get(prob_col, np.nan)) if pd.notna(row.get(prob_col, np.nan)) else np.nan
        
        edge_records.append({
            "u": u, "v": v, "edge_id": edge_id, 
            "length_m": length, "pred_prob": prob, "geometry": geom
        })

    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs="EPSG:3857")

    # Create nodes GeoDataFrame
    nodes = [
        {
            "node_id": k, "x": xy[0], "y": xy[1], 
            "geometry": Point(xy[0], xy[1])
        } 
        for k, xy in node_map.items()
    ]
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:3857")

    return nodes_gdf, edges_gdf


def _filter_by_component_length(G, nodes_gdf, edges_gdf, min_component_length_m):
    """Discard connected components with total edge length below threshold."""
    # Create simple graph for component analysis
    G_simple = nx.Graph()
    for u, v, data in G.edges(data=True):
        if not G_simple.has_edge(u, v):
            G_simple.add_edge(u, v, total_length=0.0, edge_count=0)
        G_simple[u][v]["total_length"] += data.get("length_m", 0.0)
        G_simple[u][v]["edge_count"] += 1

    # Find components that meet length requirement
    comps = list(nx.connected_components(G_simple))
    keep_nodes = set()
    
    for comp in comps:
        total_len = 0.0
        for u in comp:
            for v in G_simple[u]:
                if v in comp:
                    total_len += G_simple[u][v].get("total_length", 0.0)
        total_len = total_len / 2.0  # Undirected graph, so divide by 2
        
        if total_len >= min_component_length_m:
            keep_nodes.update(comp)

    # Filter GeoDataFrames
    nodes_gdf = nodes_gdf[nodes_gdf["node_id"].apply(lambda k: k in keep_nodes)].copy()
    edges_gdf = edges_gdf[edges_gdf.apply(
        lambda r: (r["u"] in keep_nodes and r["v"] in keep_nodes), axis=1
    )].copy()

    return nodes_gdf, edges_gdf


def _export_graph_files(nodes_gdf, edges_gdf, export_prefix):
    """Export nodes & edges to GeoJSON and GraphML (fallback to CSV if needed)."""
    try:
        # Export GeoJSON files
        nodes_gdf.to_crs(epsg=4326).to_file(f"{export_prefix}_nodes.geojson", driver="GeoJSON")
        edges_gdf.to_crs(epsg=4326).to_file(f"{export_prefix}_edges.geojson", driver="GeoJSON")
        
        # Export GraphML
        _export_graphml(nodes_gdf, edges_gdf, export_prefix)
        
    except Exception as e:
        print(f"Export failed: {e}. Attempting fallback export.")
        try:
            _fallback_export(nodes_gdf, edges_gdf, export_prefix)
        except Exception as e2:
            print(f"Fallback export also failed: {e2}. No graph outputs were written.")


def _export_graphml(nodes_gdf, edges_gdf, export_prefix):
    """Export simplified aggregated graph as GraphML (merges parallel edges)."""
    # Build aggregated graph for GraphML export
    node_pos = {
        row["node_id"]: (float(row.geometry.x), float(row.geometry.y)) 
        for _, row in nodes_gdf.iterrows()
    }
    
    G_out = nx.Graph()
    
    for _, row in edges_gdf.iterrows():
        u_key = row["u"]
        v_key = row["v"]
        
        if u_key not in node_pos or v_key not in node_pos:
            continue
            
        ux, uy = node_pos[u_key]
        vx, vy = node_pos[v_key]
        
        # Add nodes if not present
        if not G_out.has_node(str(u_key)):
            G_out.add_node(str(u_key), x=ux, y=uy)
        if not G_out.has_node(str(v_key)):
            G_out.add_node(str(v_key), x=vx, y=vy)
        
        # Add/update edge
        if G_out.has_edge(str(u_key), str(v_key)):
            # Aggregate existing edge
            prev = G_out[str(u_key)][str(v_key)].get("agg_count", 1)
            prev_len = G_out[str(u_key)][str(v_key)].get("length_m", 0.0)
            prev_prob = G_out[str(u_key)][str(v_key)].get("pred_prob", np.nan)
            
            new_len = prev_len + float(row["length_m"])
            
            if pd.notna(prev_prob) and pd.notna(row["pred_prob"]):
                new_prob = (prev_prob * prev + float(row["pred_prob"])) / (prev + 1)
            elif pd.notna(row["pred_prob"]):
                new_prob = float(row["pred_prob"])
            else:
                new_prob = prev_prob
            
            G_out[str(u_key)][str(v_key)]["length_m"] = new_len
            G_out[str(u_key)][str(v_key)]["pred_prob"] = new_prob
            G_out[str(u_key)][str(v_key)]["agg_count"] = prev + 1
        else:
            # Add new edge
            p = float(row["pred_prob"]) if pd.notna(row["pred_prob"]) else -1.0
            G_out.add_edge(
                str(u_key), str(v_key), 
                length_m=float(row["length_m"]), pred_prob=p, agg_count=1
            )

    # Write GraphML
    out_graphml = f"{export_prefix}.graphml"
    nx.write_graphml(G_out, out_graphml)


def _fallback_export(nodes_gdf, edges_gdf, export_prefix):
    """Fallback export (CSV) when GeoJSON/GraphML writing fails."""
    try:
        edges_gdf[["u", "v", "edge_id", "length_m", "pred_prob"]].to_csv(
            f"{export_prefix}_edges.csv", index=False
        )
        nodes_gdf[["node_id", "x", "y"]].to_csv(
            f"{export_prefix}_nodes.csv", index=False
        )
        print(f"Exported to CSV: {export_prefix}_edges.csv, {export_prefix}_nodes.csv")
    except Exception as e:
        print(f"Fallback export also failed: {e}")