# graph.py
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import substring
from utils import snap_coord_tuple
from config import SNAP_TOL_M, SNAP_ROUND_DECIMALS, OUT_NODES, OUT_EDGES, OUT_GRAPHML, MIN_COMPONENT_LENGTH_M
import networkx as nx
import numpy as np
import pandas as pd

def build_graph(sidewalks_gdf, connectors_gdf):
    """
    Build deterministic pedestrian graph:
    - Nodes are snapped endpoint coordinates (using SNAP_TOL_M)
    - Edges are sidewalk segments and connector segments
    - Sidewalk segments are split at connector endpoints AND at sidewalk-sidewalk intersections to ensure connectivity
    Returns (G, nodes_gdf, edges_gdf)
    """
    if (sidewalks_gdf is None or sidewalks_gdf.empty) and (connectors_gdf is None or connectors_gdf.empty):
        raise RuntimeError("No sidewalk or connector geometries provided to build graph.")

    # Prepare connector endpoints for splitting sidewalks
    connector_points = []
    if connectors_gdf is not None and not connectors_gdf.empty:
        for _, crow in connectors_gdf.iterrows():
            geom = crow.geometry
            if geom is None or geom.length == 0:
                continue
            try:
                a = Point(geom.coords[0])
                b = Point(geom.coords[-1])
                connector_points.append((a, crow.get("connector_id", None)))
                connector_points.append((b, crow.get("connector_id", None)))
            except Exception:
                continue

    # Find sidewalk-sidewalk intersection points (for T-junctions, crossroads)
    def find_sidewalk_intersections(sidewalks_gdf, tol):
        """Find intersection points between different sidewalk segments."""
        intersection_points = []
        if sidewalks_gdf is None or sidewalks_gdf.empty:
            return intersection_points
        
        print(f"  Finding sidewalk-sidewalk intersections...")
        sw_list = list(sidewalks_gdf.iterrows())
        for i, (idx_i, row_i) in enumerate(sw_list):
            for j, (idx_j, row_j) in enumerate(sw_list):
                if i >= j:
                    continue
                # Skip if same road (same origin_id)
                if "origin_id" in row_i and "origin_id" in row_j:
                    if row_i.get("origin_id") == row_j.get("origin_id"):
                        continue
                try:
                    geom_i = row_i.geometry
                    geom_j = row_j.geometry
                    if geom_i is None or geom_j is None:
                        continue
                    inter = geom_i.intersection(geom_j)
                    if inter.is_empty:
                        continue
                    if inter.geom_type == 'Point':
                        intersection_points.append((inter, f"sw_inter_{idx_i}_{idx_j}"))
                    elif inter.geom_type == 'MultiPoint':
                        for pt in inter.geoms:
                            intersection_points.append((pt, f"sw_inter_{idx_i}_{idx_j}"))
                except Exception:
                    pass
        print(f"  Found {len(intersection_points)} sidewalk intersection points")
        return intersection_points
    
    # Collect sidewalk intersection points
    sidewalk_intersection_points = find_sidewalk_intersections(sidewalks_gdf, SNAP_TOL_M)
    
    # Combine all split points (connectors + sidewalk intersections)
    all_split_points = connector_points + sidewalk_intersection_points

    def split_line_at_points(line: LineString, points: list, tol: float):
        """Split a LineString at given points (snap to line), return list of LineStrings.
        Only splits at interior positions beyond a tiny epsilon from endpoints.
        Deterministic ordering of parts from start to end.
        """
        if line is None or line.length == 0:
            return []
        # collect distances along line for points near the line
        dists = []
        for p, _cid in points:
            try:
                if p.distance(line) <= tol + 1e-9:
                    d = float(line.project(p))
                    if d > 1e-6 and d < (line.length - 1e-6):
                        dists.append(d)
            except Exception:
                continue
        if not dists:
            return [line]
        # deduplicate distances deterministically
        dists = sorted(set([round(d, 6) for d in dists]))
        parts = []
        last = 0.0
        for d in dists:
            try:
                seg = substring(line, last, d, normalized=False)
                if seg and seg.length > 0:
                    parts.append(seg)
            except Exception:
                pass
            last = d
        try:
            seg = substring(line, last, line.length, normalized=False)
            if seg and seg.length > 0:
                parts.append(seg)
        except Exception:
            pass
        return parts if parts else [line]

    # Build unified list of segments: split sidewalks at connector endpoints
    seg_records = []
    crs = None
    if sidewalks_gdf is not None and not sidewalks_gdf.empty:
        crs = sidewalks_gdf.crs
        # deterministic ordering by origin_id then side
        sw_sorted = sidewalks_gdf.copy()
        if "origin_id" in sw_sorted.columns:
            sw_sorted["origin_id_str"] = sw_sorted["origin_id"].astype(str)
            sw_sorted = sw_sorted.sort_values(["origin_id_str", "side"]).drop(columns=["origin_id_str"])
        for _, row in sw_sorted.iterrows():
            line = row.geometry
            if line is None or line.length == 0:
                continue
            parts = split_line_at_points(line, all_split_points, SNAP_TOL_M)
            for i, part in enumerate(parts):
                seg_records.append({
                    "geometry": part,
                    "type": "sidewalk",
                    "origin_id": row.get("origin_id", None),
                    "side": row.get("side", None),
                    "part_idx": i,
                })

    if connectors_gdf is not None and not connectors_gdf.empty:
        crs = connectors_gdf.crs if crs is None else crs
        conn_sorted = connectors_gdf.copy()
        if "connector_id" in conn_sorted.columns:
            conn_sorted = conn_sorted.sort_values(["connector_id"])  # deterministic
        for _, row in conn_sorted.iterrows():
            geom = row.geometry
            if geom is None or geom.length == 0:
                continue
            seg_records.append({
                "geometry": geom,
                "type": "connector",
                "connector_id": row.get("connector_id", None)
            })

    if not seg_records:
        raise RuntimeError("No segments available to build graph after processing.")

    segs = gpd.GeoDataFrame(seg_records, geometry="geometry", crs=crs)

    # create endpoints and snap deterministically
    node_map = {}  # snapped coord -> node_id
    node_rows = []
    def add_node_from_point(pt):
        x, y = pt.x, pt.y
        sx, sy = snap_coord_tuple(x, y, SNAP_TOL_M)
        key = (round(sx, SNAP_ROUND_DECIMALS), round(sy, SNAP_ROUND_DECIMALS))
        if key not in node_map:
            node_id = f"n_{len(node_map)}_{key[0]}_{key[1]}"
            node_map[key] = node_id
            node_rows.append({"node_id": node_id, "x": key[0], "y": key[1], "geometry": Point(key)})
        return node_map[key]

    edges = []
    for idx, row in segs.iterrows():
        geom = row.geometry
        if geom is None or geom.length == 0:
            continue
        # endpoints
        a = geom.coords[0]; b = geom.coords[-1]
        pa = Point(a); pb = Point(b)
        na = add_node_from_point(pa); nb = add_node_from_point(pb)
        # deterministic edge id
        eid = f"e_{idx}_{na}_{nb}"
        edges.append({"edge_id": eid, "u": na, "v": nb, "length_m": float(geom.length), "geometry": geom})

    nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=segs.crs)
    edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry", crs=segs.crs)

    # optional: prune tiny components
    G = nx.Graph()
    for _, r in nodes_gdf.iterrows():
        G.add_node(r["node_id"], x=r.geometry.x, y=r.geometry.y)
    for _, e in edges_gdf.iterrows():
        G.add_edge(e["u"], e["v"], edge_id=e["edge_id"], length_m=e["length_m"])

    # remove connected components that are too small (sum length)
    comps = list(nx.connected_components(G))
    keep_nodes = set()
    for comp in comps:
        comp_len = 0.0
        for u in comp:
            for v in G[u]:
                if v in comp:
                    comp_len += G[u][v].get("length_m", 0.0)
        comp_len = comp_len / 2.0
        if comp_len >= MIN_COMPONENT_LENGTH_M:
            keep_nodes.update(comp)

    if keep_nodes:
        nodes_gdf = nodes_gdf[nodes_gdf["node_id"].isin(keep_nodes)].copy()
        edges_gdf = edges_gdf[edges_gdf["u"].isin(keep_nodes) & edges_gdf["v"].isin(keep_nodes)].copy()

    # final deterministic sort
    nodes_gdf = nodes_gdf.sort_values("node_id").reset_index(drop=True)
    edges_gdf = edges_gdf.sort_values("edge_id").reset_index(drop=True)

    return G, nodes_gdf, edges_gdf

def export_graph(nodes_gdf, edges_gdf, out_nodes, out_edges, out_graphml):
    try:
        nodes_gdf.to_crs(epsg=4326).to_file(out_nodes, driver="GeoJSON")
        edges_gdf.to_crs(epsg=4326).to_file(out_edges, driver="GeoJSON")
    except Exception as e:
        print("GeoJSON export failed:", e)
    # write graphml
    try:
        G = nx.Graph()
        for _, n in nodes_gdf.iterrows():
            G.add_node(n["node_id"], x=float(n.geometry.x), y=float(n.geometry.y))
        for _, e in edges_gdf.iterrows():
            G.add_edge(e["u"], e["v"], edge_id=e["edge_id"], length_m=float(e["length_m"]))
        nx.write_graphml(G, out_graphml)
    except Exception as e:
        print("GraphML export failed:", e)
