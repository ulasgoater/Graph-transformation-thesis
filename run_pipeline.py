# run_pipeline.py
from config import *
from io_utils import load_autograph_csv, load_point_csv
from classify import classify_autograph
from sidewalks import make_sidewalks
from crossings import insert_crossings
from graph import build_graph, export_graph
import geopandas as gpd

def main(run_test_subset=True):
    print("Loading autograph...")
    gdf_auto = load_autograph_csv(AUTOGRAPH_CSV)
    print("Autograph segments:", len(gdf_auto))

    # optional: load park polygons if you have them (not provided in your CSVs)
    gdf_parks = None

    # load crossings points
    try:
        gdf_cross = load_point_csv(CROSSINGS_CSV)
        print("Crossings loaded:", len(gdf_cross))
    except Exception as e:
        print("Crossings load failed or missing:", e)
        gdf_cross = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_auto.crs)
    
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
    main(run_test_subset=True)
