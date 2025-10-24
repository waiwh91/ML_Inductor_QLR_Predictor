import numpy as np
import pandas as pd

def freq_adder(data, output):
    data = np.repeat(data, 5, axis=0)
    x, y = data.shape

    test_data = np.zeros([x, y + 1])

    for i in range(1, data.shape[0] + 1):
        if i % 5 == 1:
            test_data[i - 1] = np.append(data[i - 1], np.array([1]), axis=0)
        elif i % 5 == 0:
            test_data[i - 1] = np.append(data[i - 1], np.array([100]), axis=0)
        elif i % 5 == 2:
            test_data[i - 1] = np.append(data[i - 1], np.array([25.75]), axis=0)
        elif i % 5 == 3:
            test_data[i - 1] = np.append(data[i - 1], np.array([50.5]), axis=0)
        elif i % 5 == 4:
            test_data[i - 1] = np.append(data[i - 1], np.array([75.25]), axis=0)



    output_pd = pd.DataFrame(
        {"tCu": test_data[:, 0], "wCu": test_data[:, 1], "tLam": test_data[:, 2], "nLam": test_data[:, 3],
         "aln": test_data[:, 4], "tsu": test_data[:, 5], "freq": test_data[:, 6]})
    output_pd.to_csv(output, index=False)

    return test_data




