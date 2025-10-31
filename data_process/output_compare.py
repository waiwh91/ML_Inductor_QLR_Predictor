import numpy as np
import pandas as pd

def compare(predicted_path, real_path, output_path):
    predicted_csv = pd.read_csv(predicted_path).to_numpy()

    real_csv = pd.read_csv(real_path).to_numpy()


    ####### 50% compare
    x, y = real_csv.shape
    print(x, y)

    total_data = np.zeros((x, y+3))
    # total_data_50 = []

    i = 0

    for predicted_data in predicted_csv:
        for simulated_data in real_csv:
            if (predicted_data[:7] == simulated_data[:7]).all():

                total_data[i] = np.append(simulated_data, predicted_data[7:10])
                # total_data_50.append(simulated_data + predicted_data[7:10])
                i += 1
    df = pd.DataFrame({"tCu": total_data[:,0],"wCu": total_data[:,1],"tLam":total_data[:,2], "nLam": total_data[:,3], "aln": total_data[:,4],
                          "tsu": total_data[:,5], "freq": total_data[:,6], "real_Q":total_data[:,7], "real_R": total_data[:,8],"real_L": total_data[:,9],
                          "pre_Q":total_data[:,10], "pre_R":total_data[:,11], "pre_L":total_data[:,12]})
    df.to_csv(output_path, index=False)
