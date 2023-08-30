def calculate_column_averages(data):
    num_columns = len(data[0])
    column_sums = [0] * num_columns
    
    for row in data:
        for i, value in enumerate(row):
            column_sums[i] += value
    
    num_rows = len(data)
    column_averages = [(sum_value / num_rows) for sum_value in column_sums]
    
    result = []
    for _ in range(num_rows):
        result.append(tuple(column_averages))
    
    return result

data = [(63, 736, 83, 73), (63, 735, 735, 892),(63, 735, 735, 892), (63, 735, 735, 892), (63, 735, 735, 892),  (73, 836, 383, 83)]
averages = calculate_column_averages(data)
print(averages)
