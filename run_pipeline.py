# run_pipeline.py
from config import *
from data_fetcher import (
    fetch_autograph_data,
    fetch_crossings_data,
    fetch_amenities_data,
    fetch_pedestrian_data,
)
from classify import classify_autograph
from sidewalks import make_sidewalks
from crossings import insert_crossings
from graph import build_graph, export_graph


def get_bbox_from_user():
    """Prompt user for bounding box coordinates."""
    print("\nEnter bounding box coordinates for the area:")
    print("Format: south, west, north, east (latitude and longitude in decimal degrees)")
    print("Example for Milano: 45.386, 9.040, 45.535, 9.278")
    
    while True:
        try:
            coords_input = input("\nCoordinates: ").strip()
            coords = [float(x.strip()) for x in coords_input.split(",")]
            
            if len(coords) != 4:
                print("Error: Please enter exactly 4 values (south, west, north, east)")
                continue
            
            south, west, north, east = coords
            
            if not (-90 <= south < north <= 90):
                print("Error: Invalid latitude values. South must be < North, both between -90 and 90")
                continue
            
            if not (-180 <= west < east <= 180):
                print("Error: Invalid longitude values. West must be < East, both between -180 and 180")
                continue
            
            return coords
        except ValueError:
            print("Error: Please enter valid numeric coordinates")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            exit(0)


def main(run_test_subset=True, bbox=None, prompt_bbox=False):
    """Run the pedestrian graph pipeline using Overpass API.
    
    Args:
        run_test_subset: If True, extract only a 1km x 1km test area
        bbox: Custom bounding box [south, west, north, east], uses Milano default if None
        prompt_bbox: If True, interactively prompt user for coordinates
    """
    if prompt_bbox:
        bbox = get_bbox_from_user()
    
    if bbox:
        print(f"Fetching data from Overpass API for bbox: {bbox} …")
    else:
        print("Fetching data from Overpass API (default Milano bbox) …")
    
    gdf_auto = fetch_autograph_data(bbox)
    gdf_cross = fetch_crossings_data(bbox)
    gdf_amenities = fetch_amenities_data(bbox)
    gdf_pedestrian = fetch_pedestrian_data(bbox)
    
    # Export raw Overpass query results
    print("\nExporting raw Overpass query results...")
    OUT_DIR.mkdir(exist_ok=True)
    gdf_auto.to_file(OUT_DIR / "autograph_raw.geojson", driver="GeoJSON")
    gdf_cross.to_file(OUT_DIR / "crossings_raw.geojson", driver="GeoJSON")
    gdf_amenities.to_file(OUT_DIR / "amenities_raw.geojson", driver="GeoJSON")
    gdf_pedestrian.to_file(OUT_DIR / "pedestrian_raw.geojson", driver="GeoJSON")
    print(f"Raw data exported to: {OUT_DIR}")

    print("\nAutograph segments:", len(gdf_auto))
    print("Crossings loaded:", len(gdf_cross))

    # optional: load park polygons if you have them (not provided in your CSVs)
    gdf_parks = None

    if gdf_cross.empty:
        print("WARNING: No crossing points provided. Graph will have no crossing connectors.")
        print("         Pedestrian graph may be disconnected between road sides.")

    # 1) classify deterministically (with proximity buffer for possibly-pedestrian roads)
    gdf_classed = classify_autograph(gdf_auto, gdf_parks, proximity_buffer_m=PROXIMITY_BUFFER_M)

    # optionally take a test subset (spatial)
    if run_test_subset:
        if len(gdf_classed) == 0:
            raise RuntimeError("No autograph rows")
        # pick first centroid and take bounding box of 1km
        c = gdf_classed.geometry.centroid.iloc[0]
        minx, miny, maxx, maxy = c.x - 1000, c.y - 1000, c.x + 1000, c.y + 1000
        gdf_classed = gdf_classed[gdf_classed.geometry.centroid.x.between(minx, maxx) & gdf_classed.geometry.centroid.y.between(miny, maxy)].copy()
        gdf_cross = gdf_cross[gdf_cross.geometry.x.between(minx, maxx) & gdf_cross.geometry.y.between(miny, maxy)].copy()
        print("TEST subset sizes:", len(gdf_classed), len(gdf_cross))

    # keep only segments deemed pedestrian by rules
    ped_segs = gdf_classed[gdf_classed["is_pedestrian_road"] == True].copy()
    print("Pedestrian segments (by rule):", len(ped_segs))

    # 2) create sidewalks
    if "origin_id" in ped_segs.columns:
        keep_cols_list = ["origin_id"]
    else:
        keep_cols_list = None
    sidewalks_gdf = make_sidewalks(ped_segs, SIDEWALK_OFFSET_M, keep_cols=keep_cols_list)
    # ensure proper crs
    sidewalks_gdf = sidewalks_gdf.set_crs(gdf_auto.crs, allow_override=True)
    print("Sidewalk segments generated:", len(sidewalks_gdf))

    # 3) insert crossings deterministically
    connectors_gdf = insert_crossings(sidewalks_gdf, gdf_cross, CROSSING_SEARCH_M)
    print("Connectors generated:", len(connectors_gdf))

    # 4) build graph and export
    G, nodes_gdf, edges_gdf = build_graph(sidewalks_gdf, connectors_gdf)
    print("Nodes:", len(nodes_gdf), "Edges:", len(edges_gdf))
    export_graph(nodes_gdf, edges_gdf, OUT_NODES, OUT_EDGES, OUT_GRAPHML)
    print("Exported to:", OUT_NODES, OUT_EDGES, OUT_GRAPHML)
    print("Done.")

if __name__ == "__main__":
    main(run_test_subset=False, prompt_bbox=True)
