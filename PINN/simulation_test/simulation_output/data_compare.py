import pandas as pd
import numpy as np

# data_processor = data_processor.data_processor("RLQ")
# data_processor.process_dir()

pre_data = pd.read_csv('~/ML_inductor/simulation_test/predicted_output.csv').to_numpy()
simu_data = pd.read_csv('~/ML_inductor/simulation_test/simulation_output/total_data.csv').to_numpy()

x, y = pre_data.shape
print(x, y)

compare_data = np.zeros([x, y + 3])

for i in range(len(pre_data)):
    for j in range(len(simu_data)):
        if (pre_data[i][:7] == simu_data[j][:7]).all():
            # print(pre_data[i][:7], simu_data[j][:7])
            compare_data = np.append(simu_data[:,:], pre_data[:,7:10], axis=1)
            break
    
print(compare_data)

