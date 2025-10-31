# analyze_graph.py
"""
Compute quality metrics for the generated pedestrian graph.
Use these metrics for thesis validation and comparison.
"""
import geopandas as gpd
import networkx as nx
from pathlib import Path
from config import OUT_NODES, OUT_EDGES, OUT_GRAPHML

def load_graph():
    """Load the generated graph from output files."""
    if not OUT_GRAPHML.exists():
        raise FileNotFoundError(f"Graph file not found: {OUT_GRAPHML}")
    
    G = nx.read_graphml(OUT_GRAPHML)
    nodes_gdf = gpd.read_file(OUT_NODES)
    edges_gdf = gpd.read_file(OUT_EDGES)
    
    return G, nodes_gdf, edges_gdf

def compute_metrics(G, nodes_gdf, edges_gdf):
    """Compute comprehensive graph quality metrics."""
    metrics = {}
    
    # Basic counts
    metrics['num_nodes'] = len(nodes_gdf)
    metrics['num_edges'] = len(edges_gdf)
    
    # Total network length
    edges_gdf_m = edges_gdf.to_crs(epsg=3857)  # Convert to meters
    total_length_m = edges_gdf_m.geometry.length.sum()
    metrics['total_length_m'] = total_length_m
    metrics['total_length_km'] = total_length_m / 1000.0
    
    # Average edge length
    metrics['avg_edge_length_m'] = edges_gdf_m.geometry.length.mean()
    metrics['min_edge_length_m'] = edges_gdf_m.geometry.length.min()
    metrics['max_edge_length_m'] = edges_gdf_m.geometry.length.max()
    
    # Graph connectivity
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    metrics['num_connected_components'] = nx.number_connected_components(G_undirected)
    
    # Largest component
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    metrics['largest_component_nodes'] = len(largest_cc)
    metrics['largest_component_pct'] = (len(largest_cc) / len(G_undirected.nodes)) * 100 if len(G_undirected.nodes) > 0 else 0
    
    # Node degrees
    degrees = [deg for node, deg in G_undirected.degree()]
    metrics['avg_node_degree'] = sum(degrees) / len(degrees) if degrees else 0
    metrics['max_node_degree'] = max(degrees) if degrees else 0
    metrics['min_node_degree'] = min(degrees) if degrees else 0
    
    # Count terminal nodes (degree 1)
    metrics['num_terminal_nodes'] = sum(1 for d in degrees if d == 1)
    
    # Count intersection nodes (degree > 2)
    metrics['num_intersections'] = sum(1 for d in degrees if d > 2)
    
    # Edge types (if available)
    if 'type' in edges_gdf.columns:
        edge_types = edges_gdf['type'].value_counts().to_dict()
        metrics['edge_types'] = edge_types
        metrics['num_sidewalk_edges'] = edge_types.get('sidewalk', 0)
        metrics['num_connector_edges'] = edge_types.get('connector', 0)
    
    # Spatial extent
    bounds = nodes_gdf.total_bounds  # minx, miny, maxx, maxy
    nodes_gdf_m = nodes_gdf.to_crs(epsg=3857)
    bounds_m = nodes_gdf_m.total_bounds
    width_m = bounds_m[2] - bounds_m[0]
    height_m = bounds_m[3] - bounds_m[1]
    area_km2 = (width_m * height_m) / 1_000_000
    
    metrics['bounds'] = bounds
    metrics['width_m'] = width_m
    metrics['height_m'] = height_m
    metrics['area_km2'] = area_km2
    
    # Density metrics
    if area_km2 > 0:
        metrics['node_density_per_km2'] = metrics['num_nodes'] / area_km2
        metrics['edge_density_per_km2'] = metrics['num_edges'] / area_km2
        metrics['length_density_km_per_km2'] = metrics['total_length_km'] / area_km2
        if 'num_connector_edges' in metrics:
            metrics['crossing_density_per_km2'] = metrics['num_connector_edges'] / area_km2
    
    return metrics

def print_metrics(metrics):
    """Print metrics in a formatted report."""
    print("\n" + "=" * 70)
    print("PEDESTRIAN GRAPH QUALITY METRICS")
    print("=" * 70)
    
    print("\n📊 BASIC STATISTICS")
    print(f"  Nodes:                    {metrics['num_nodes']:,}")
    print(f"  Edges:                    {metrics['num_edges']:,}")
    print(f"  Total length:             {metrics['total_length_km']:.2f} km ({metrics['total_length_m']:.1f} m)")
    
    print("\n📏 EDGE STATISTICS")
    print(f"  Average edge length:      {metrics['avg_edge_length_m']:.2f} m")
    print(f"  Min edge length:          {metrics['min_edge_length_m']:.2f} m")
    print(f"  Max edge length:          {metrics['max_edge_length_m']:.2f} m")
    
    if 'edge_types' in metrics:
        print(f"\n  Edge types:")
        for etype, count in metrics['edge_types'].items():
            print(f"    - {etype:15s}   {count:,}")
    
    print("\n🔗 CONNECTIVITY")
    print(f"  Connected components:     {metrics['num_connected_components']}")
    print(f"  Largest component:        {metrics['largest_component_nodes']:,} nodes ({metrics['largest_component_pct']:.1f}%)")
    print(f"  Average node degree:      {metrics['avg_node_degree']:.2f}")
    print(f"  Max node degree:          {metrics['max_node_degree']}")
    print(f"  Terminal nodes (deg=1):   {metrics['num_terminal_nodes']:,}")
    print(f"  Intersections (deg>2):    {metrics['num_intersections']:,}")
    
    print("\n🗺️  SPATIAL EXTENT")
    print(f"  Bounds (lon/lat):         {metrics['bounds']}")
    print(f"  Width:                    {metrics['width_m']:.1f} m")
    print(f"  Height:                   {metrics['height_m']:.1f} m")
    print(f"  Area:                     {metrics['area_km2']:.3f} km²")
    
    if 'node_density_per_km2' in metrics:
        print("\n📍 DENSITY METRICS")
        print(f"  Nodes per km²:            {metrics['node_density_per_km2']:.1f}")
        print(f"  Edges per km²:            {metrics['edge_density_per_km2']:.1f}")
        print(f"  Length density:           {metrics['length_density_km_per_km2']:.2f} km/km²")
        if 'crossing_density_per_km2' in metrics:
            print(f"  Crossings per km²:        {metrics['crossing_density_per_km2']:.1f}")
    
    print("\n" + "=" * 70)

def main():
    """Main analysis entry point."""
    try:
        print("Loading generated graph...")
        G, nodes_gdf, edges_gdf = load_graph()
        
        print("Computing metrics...")
        metrics = compute_metrics(G, nodes_gdf, edges_gdf)
        
        print_metrics(metrics)
        
        return metrics
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
