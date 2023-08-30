import networkx as nx
import matplotlib.pyplot as plt

# Sample data: (accuracy, runtime, iterations, size)
data = [
    (0.95, 10, 100, 50),
    (0.85, 15, 200, 60),
    (0.92, 12, 150, 55),
    (0.88, 18, 180, 70),
    (0.98, 8, 80, 45),
]

print(data)
# Create a directed graph
G = nx.DiGraph()

# Add nodes with attributes
for i, (accuracy, runtime, iterations, size) in enumerate(data):
    G.add_node(i, accuracy=accuracy, runtime=runtime, iterations=iterations, size=size)

# Add edges based on iterations and size
for source in G.nodes():
    for target in G.nodes():
        if source != target:
            source_iter = G.nodes[source]['iterations']
            target_size = G.nodes[target]['size']
            G.add_edge(source, target, weight=source_iter / target_size)

# Get node and link data arrays for visualization
nodes_data = [{'id': node, **data} for node, data in G.nodes(data=True)]
links_data = [{'source': source, 'target': target, 'weight': data['weight']} for source, target, data in G.edges(data=True)]

# Visualization
pos = nx.spring_layout(G)  # Layout algorithm
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')
edge_labels = {(source, target): f"{data['weight']:.2f}" for source, target, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.show()

# Print nodes and links data arrays
print("Nodes Data:")
for node_data in nodes_data:
    print(node_data)

print("\nLinks Data:")
for link_data in links_data:
    print(link_data)
