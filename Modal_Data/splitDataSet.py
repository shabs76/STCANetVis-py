import numpy as np
import math

def load_csv_to_numpy(csv_file_path):
    # Load the CSV file into a NumPy array
    lenPara = 50
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=None, encoding=None)
    if len(data) <= 500:
        return  {
            'state': 'error',
            'data': 'dataset is too short. Should have more than 400 rows'
        }
    remlen = len(data) - lenPara
    addVal = remlen/lenPara
    decimal_part = addVal - math.floor(addVal)
    valc = 0
    groups = []
    if decimal_part > 0.05:
        valc = remlen%lenPara
    # Return the first 200 rows
    startAddv = lenPara
    if valc > 10:
        wholN = math.floor(addVal)
        if wholN >= 1:
           for xv in range(lenPara):
               grow = 0
               if startAddv <= valc:
                   grow = 1
               frow = int((xv*wholN) + lenPara + (grow* (valc - startAddv)))
               groups.append(data[:frow])
               startAddv = startAddv - 1
        else:
            for xvc in range(lenPara):
                grows = 0
                if startAddv <= valc:
                   grows = 1
                frowc = int(xvc + (grows* (valc - startAddv)) + 1)
                groups.append(data[:frowc])
                startAddv = startAddv - 1
    else:
        wholNT = math.floor(addVal)
        if wholNT >= 1:
           for xvT in range(lenPara):
               frowT = int((xvT*wholNT) + lenPara)
               groups.append(data[:frowT])
        else:
            for xvcT in range(lenPara):
                frowcT = int(xvcT  + 1)
                groups.append(data[:frowcT])

    # if groups:
    #     for idx, group in enumerate(groups, start=1):
    #         print(f"Group {idx} Size: {group.shape[0]}")
    # else:
    #     print("Dataset has less than 200 rows. Cannot divide into groups.")
    
    # return groups
    groups.append(data)
    return {
        'state': 'success',
        'data': groups,
        'wholeSh': len(data)
    }

# csv_file_path = 'datasets/waveData.csv'  # Replace with the path to your CSV file
# result = load_csv_to_numpy(csv_file_path)

# print(len(result))


# # datasets/waveData.csv