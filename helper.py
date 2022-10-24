import numpy as np

def OHE(arr_of_ints):
    unique_num = len(np.unique(arr_of_ints))
    all_vectors = []
    
    for i in range(len(arr_of_ints)):
        val = arr_of_ints[i]
        temp = np.zeros(unique_num)
        temp[val] = 1
        all_vectors.append(temp)
    
    return all_vectors