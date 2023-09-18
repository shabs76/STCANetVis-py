import networkx as nx
import matplotlib.pyplot as plt

data = [
    [0.8, 20, 100, 200],
    [0.7, 40, 20, 200],
    [0.4, 10, 10, 200],
    [0.3, 20, 100, 30],
    [0.4, 23, 20, 20],
    [0.1, 90, 10, 40]
]

# Create a directed graph
G = nx.DiGraph()

# Create nodes and add them to the graph based on size and iterations
nodes = {}
for d in data:
    size = d[3]
    iterations = d[2]
    key = (size, iterations)
    if key not in nodes:
        nodes[key] = {
            'size': size,
            'iterations': iterations,
            'data': []
        }
    nodes[key]['data'].append(d)

for key, node_data in nodes.items():
    G.add_node(key, **node_data)

# Create links based on size and iterations
for key1 in nodes:
    for key2 in nodes:
        if key1 != key2:
            if nodes[key1]['size'] == nodes[key2]['size']:
                G.add_edge(key1, key2)
            elif nodes[key1]['iterations'] == nodes[key2]['iterations']:
                G.add_edge(key1, key2)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
edge_labels = {(n1, n2): f"{nodes[n1]['size']}/{nodes[n1]['iterations']}" for n1, n2 in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Network Graph")
plt.show()

# Display nodes and links data
print("Nodes Data:")
for key, node_data in nodes.items():
    print(f"Node {key}:")
    for d in node_data['data']:
        print(d)
    print("---")

print("Links Data:")
for edge in G.edges():
    print(f"Link between {edge[0]} and {edge[1]}")

