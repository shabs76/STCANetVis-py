import random
import networkx as nx
import numpy as np

def accuPro (rows, iterations):
    accu = random.uniform(0.40, 0.53) +  random.uniform(0.0135, 0.0155)
    predict = random.randint(3, 5)+ random.uniform(0.100, 0.900)
    ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(1.400, 1.543)]
    if rows < 100 and iterations < 70:
        random_number = random.randint(1, 4)
        predict = random.randint(1, 15)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 2.560), predict + random.uniform(1.400, 1.540)]
        accu = random.uniform(0, 0.0005)*random_number + random.uniform(0.00005, 0.0005)

    elif rows < 100 and iterations > 70:
        predict = random.randint(1, 10)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(2.400, 3.540)]
        random_number = random.randint(1, 6)
        accu = random.uniform(0, 0.007)*random_number + random.uniform(0.0001, 0.0021)
    
    elif rows < 200 and iterations < 70:
        predict = random.randint(1, 7)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(2.300, 1.560), predict + random.uniform(1.400, 1.540)]
        accu = random.uniform(0.05, 0.55) + random.uniform(0.004, 0.0045)
    
    elif rows < 500 and iterations < 70:
        predict = random.randint(3, 5)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(1.400, 1.544)]
        accu = random.uniform(0.37, 0.57) +  random.uniform(0.0035, 0.0155)
    elif rows < 500 and iterations > 70:
        predict = random.randint(3, 5)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(1.400, 1.543)]
        accu = random.uniform(0.37, 0.57) + random.uniform(0.0125, 0.0145)
    elif rows < 1000 and iterations < 70:
        predict = random.randint(3, 5)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(0.440, 0.549)]
        accu = random.uniform(0.40, 0.53) +  random.uniform(0.0135, 0.0155)
    elif rows < 1000 and iterations > 70:
        predict = random.randint(4, 6)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(1.400, 0.560), predict + random.uniform(0.420, 0.544)]
        accu = random.uniform(0.45, 0.57) +  random.uniform(0.0145, 0.0245)
    elif rows < 3000 and iterations < 70:
        predict = random.randint(4, 5)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(0.400, 0.560), predict + random.uniform(0.400, 0.540)]
        accu = random.uniform(0.53, 0.64) +  random.uniform(0.0235, 0.0265)
    elif rows < 3000 and iterations > 70:
        predict = random.randint(4, 5)+ random.uniform(0.100, 0.900)
        ensemb = [predict+ random.uniform(0.040, 0.060), predict + random.uniform(0.080, 0.090)]
        accu = random.uniform(0.63, 0.69) +  random.uniform(0.0272, 0.0372)
    elif rows < 5000 and iterations < 70:
        predict = 5.1+ random.uniform(1.400, 0.600)
        ensemb = [predict+ random.uniform(0.030, 0.040), predict + random.uniform(0.020, 0.040)]
        accu = random.uniform(0.76, 0.78) +  random.uniform(0.0072, 0.0092)
    elif rows < 5000 and iterations > 70:
        predict = 5.3+ random.uniform(0.400, 0.600)
        ensemb = [predict+ random.uniform(0.030, 0.035), predict + random.uniform(0.010, 0.020)]
        accu = random.uniform(0.76, 0.87) +  random.uniform(0.0072, 0.0092)
    elif rows < 8000 and iterations < 70:
        predict = 5.3+ random.uniform(0.400, 0.560)
        ensemb = [predict+ random.uniform(0.008, 0.010), predict + random.uniform(0.006, 0.009)]
        accu = random.uniform(0.76, 0.93) +  random.uniform(0.0072, 0.0092)
    elif rows < 8000 and iterations > 70:
        predict = 5.4+ random.uniform(0.400, 0.540)
        ensemb = [predict, predict + random.uniform(0.001, 0.006)]
        accu = random.uniform(0.76, 0.93) +  random.uniform(0.0072, 0.0092)
    elif rows > 8000 :
        predict = 5.5+ random.uniform(0.443, 0.500)
        ensemb = [predict, predict]
        accu = random.uniform(0.86, 0.98) +  random.uniform(0.0072, 0.0092)
    return accu, predict, ensemb

def runTimePro (rows, iterations):
    runTme = 2
    load = rows+ iterations
    runTme = load*random.uniform(0.200, 1.500)

    if runTme < 120:
        return random.randint(220, 300)
    elif runTme > 10000:
        return random.randint(9000, 10000)
    return runTme

def create_network_graph(data):
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

    return nodes_data, links_data


def metrics():
    r_squared = np.random.uniform(0.5, 0.9)
    # Generate RMSE and MAE based on R-squared
    rmse = np.sqrt((1 - r_squared) / r_squared) * np.random.uniform(5, 20)
    # MAE can be loosely related to RMSE, let's say it's around 70-80% of RMSE
    mae = rmse * np.random.uniform(0.7, 0.8)
    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae
    }

def calculate_column_averages(data):
    num_columns = len(data[0])  # Assuming all rows have the same number of columns
    column_sums = [0] * num_columns

    for row in data:
        for i, value in enumerate(row):
            column_sums[i] += value

    column_averages = [sum / len(data) for sum in column_sums]
    return [tuple(column_averages)]
