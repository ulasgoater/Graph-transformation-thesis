import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Load your GraphML file
script_dir = Path(__file__).parent
G = nx.read_graphml(script_dir / "ped_graph.graphml")

# Choose a layout (spring layout works well for general graphs)
pos = nx.spring_layout(G, seed=42)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(
    G,
    pos,
    with_labels=False,   # change to True if you want node labels
    node_size=20,
    edge_color="gray"
)

plt.show()