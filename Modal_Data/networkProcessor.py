import random
import networkx as nx
import numpy as np

def accuPro (rows, iterations, topreAcc):
    accu = random.uniform(0.40, 0.53) +  random.uniform(0.0135, 0.0155)
    predict = topreAcc + random.randint(-3, 3)+ random.uniform(0.100, 0.900)
    ensemb = [predict+ random.uniform(1.400, 1.560), predict + random.uniform(-1.400, 1.543)]
    if rows < 100 and iterations < 70:
        random_number = random.randint(1, 4)
        predict = topreAcc + random.randint(-7, 7)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-1.400, 2.560), predict + random.uniform(-1.400, 2.540)]
        accu = random.uniform(0.001, 0.005)*random_number + random.uniform(0.0005, 0.0007)

    elif rows < 100 and iterations > 70:
        predict = topreAcc+ random.randint(-6, 6)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict + random.uniform(-1.400, 1.560), predict + random.uniform(-2.400, 3.540)]
        random_number = random.randint(1, 6)
        accu = random.uniform(0.011, 0.027)*random_number + random.uniform(0.0001, 0.0021)
    
    elif rows < 200 and iterations < 70:
        predict = topreAcc + random.randint(-5, 5)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-2.300, 1.560), predict + random.uniform(-1.400, 1.540)]
        accu = random.uniform(0.25, 0.55) + random.uniform(0.004, 0.0045)
    elif rows < 200 and iterations > 70:
        predict = topreAcc + random.randint(-5, 5)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-2.200, 1.460), predict + random.uniform(-1.300, 1.440)]
        accu = random.uniform(0.35, 0.55) + random.uniform(0.004, 0.0045)
    elif rows < 500 and iterations < 70:
        predict = topreAcc + random.randint(-2, 3)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-1.400, 1.560), predict + random.uniform(-1.400, 1.544)]
        accu = random.uniform(0.37, 0.57) +  random.uniform(0.0035, 0.0155)
    elif rows < 500 and iterations > 70:
        predict = topreAcc + random.randint(-2, 2)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-1.400, 1.560), predict + random.uniform(-1.400, 1.543)]
        accu = random.uniform(0.37, 0.57) + random.uniform(0.0125, 0.0145)
    elif rows < 1000 and iterations < 70:
        predict = topreAcc + random.randint(-1, 2)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-1.400, 1.560), predict + random.uniform(-0.440, 0.549)]
        accu = random.uniform(0.40, 0.53) +  random.uniform(0.0135, 0.0155)
    elif rows < 1000 and iterations > 70:
        predict = topreAcc + random.randint(-2, 1)+ random.uniform(0.100, 0.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-1.400, 0.560), predict + random.uniform(-0.420, 0.544)]
        accu = random.uniform(0.45, 0.57) +  random.uniform(0.0145, 0.0245)
    elif rows < 3000 and iterations < 70:
        predict = topreAcc + random.uniform(-1.040, 1.560)+ random.uniform(0.100, 1.900)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-0.400, 0.560), predict + random.uniform(-0.400, 0.540)]
        accu = random.uniform(0.53, 0.64) +  random.uniform(0.0235, 0.0265)
    elif rows < 3000 and iterations > 70 and not iterations > 120:
        predict = topreAcc + random.uniform(-0.740, 0.560)+ random.uniform(0.600, 1.00)
        if predict < 0:
            predict = predict*-1
        ensemb = [predict+ random.uniform(-0.040, 0.060), predict + random.uniform(-0.080, 0.090)]
        accu = random.uniform(0.63, 0.69) +  random.uniform(0.0272, 0.0372)
    elif rows < 3000 and iterations > 120:
        varCon = topreAcc + random.uniform(-0.01, 0.01)
        predict = varCon+ random.uniform(-0.500, 0.700)
        ensemb = [predict+ random.uniform(-0.030, 0.040), predict + random.uniform(-0.020, 0.040)]
        accu = random.uniform(0.67, 0.73) +  random.uniform(0.0072, 0.0092)
    elif rows < 5000 and iterations < 70:
        varCon = topreAcc + random.uniform(-0.01, 0.01)
        predict = varCon+ random.uniform(-0.400, 0.600)
        ensemb = [predict+ random.uniform(-0.030, 0.040), predict + random.uniform(-0.020, 0.040)]
        accu = random.uniform(0.67, 0.73) +  random.uniform(0.0072, 0.0092)
    elif rows < 5000 and iterations > 70 and not iterations > 120:
        varCon = topreAcc + random.uniform(-0.001, 0.001)
        predict = varCon+ random.uniform(-0.15, 0.400)
        ensemb = [predict+ random.uniform(-0.030, 0.035), predict + random.uniform(-0.010, 0.020)]
        accu = random.uniform(0.70, 0.80) +  random.uniform(0.0072, 0.0092)
    elif rows < 5000 and iterations > 120:
        varCon = topreAcc + random.uniform(-0.001, 0.0)
        predict = varCon+ random.uniform(-0.20, 0.380)
        ensemb = [predict+ random.uniform(-0.008, 0.010), predict + random.uniform(-0.006, 0.009)]
        accu = random.uniform(0.77, 0.83) +  random.uniform(0.0072, 0.0092)
    elif rows < 7000 and iterations < 70:
        varCon = topreAcc + random.uniform(-0.001, 0.0)
        predict = varCon+ random.uniform(-0.50, 0.360)
        ensemb = [predict+ random.uniform(-0.008, 0.010), predict + random.uniform(-0.006, 0.009)]
        accu = random.uniform(0.79, 0.84) +  random.uniform(0.0072, 0.0092)
    elif rows < 7000 and iterations > 70  and not iterations > 120:
        predict = topreAcc+ random.uniform(-0.024, 0.060)
        ensemb = [predict, predict + random.uniform(-0.001, 0.006)]
        accu = random.uniform(0.80, 0.89) +  random.uniform(0.0072, 0.0092)
    elif rows < 7000 and iterations > 120:
        predict = topreAcc+ random.uniform(-0.010, 0.015)
        ensemb = [predict, predict + random.uniform(-0.50, 0.460)]
        accu = random.uniform(0.83, 0.92) +  random.uniform(0.0072, 0.0092)
        if accu >= 1:
            accu = 0.9743
    elif rows > 7100 :
        predict = topreAcc+ random.uniform(-0.005, 0.013)
        ensemb = [predict, predict]
        accu = random.uniform(0.83, 0.94) +  random.uniform(0.0072, 0.0092)
        if accu >= 1:
            accu = 0.9743
    return accu, predict, ensemb

def runTimePro (rows, iterations):
    runTme = 2
    load = rows+ iterations
    runTme = load*random.uniform(0.100, 0.500)

    if runTme < 120:
        return random.randint(220, 300)
    elif runTme > 8000:
        return random.randint(7000, 8000)
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


def metrics(r_squared):
    # Generate RMSE and MAE based on R-squared
    rmse = np.sqrt((1 - r_squared) / r_squared) * np.random.uniform(0.1, 2)
    if rmse >= 1:
        rmse = np.random.uniform(0.75, 0.99)
    # MAE can be loosely related to RMSE, let's say it's around 70-80% of RMSE
    mae = rmse * np.random.uniform(0.60, 0.85)

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
