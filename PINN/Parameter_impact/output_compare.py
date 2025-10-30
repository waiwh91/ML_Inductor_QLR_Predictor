import numpy as np
import pandas as pd

def output_compare_50_100(model_path):
    predicted_50 = pd.read_csv(f"../{model_path}/predicted_csv/predicted_50.csv").to_numpy()
    predicted_100 = pd.read_csv(f"../{model_path}/predicted_csv/predicted_100.csv").to_numpy()




    simulated_50 = pd.read_csv(f"../{model_path}/simulator_output_data/csv_50.csv").to_numpy()
    simulated_100 = pd.read_csv(f"../{model_path}/simulator_output_data/csv_100.csv").to_numpy()

    ####### 50% compare
    x_50, y_50 = simulated_50.shape
    print(x_50, y_50)

    total_data_50 = np.zeros((x_50, y_50+3))
    # total_data_50 = []

    i = 0

    for predicted_data in predicted_50:
        for simulated_data in simulated_50:
            if (predicted_data[:7] == simulated_data[:7]).all():

                total_data_50[i] = np.append(simulated_data, predicted_data[7:10])
                # total_data_50.append(simulated_data + predicted_data[7:10])
                i += 1
    df_50 = pd.DataFrame({"tCu": total_data_50[:,0],"wCu": total_data_50[:,1],"tLam":total_data_50[:,2], "nLam": total_data_50[:,3], "aln": total_data_50[:,4],
                          "tsu": total_data_50[:,5], "freq": total_data_50[:,6], "real_Q":total_data_50[:,7], "real_R": total_data_50[:,8],"real_L": total_data_50[:,9],
                          "pre_Q":total_data_50[:,10], "pre_R":total_data_50[:,11], "pre_L":total_data_50[:,12]})
    df_50.to_csv(f"{model_path}/Data_compare/compare_50.csv", index=False)




    ####### 100% compare
    x_100, y_100 = simulated_100.shape
    print(x_100, y_100)

    total_data_100 = np.zeros((x_100, y_100+3))
    # total_data_50 = []

    i = 0

    for predicted_data in predicted_100:
        for simulated_data in simulated_100:
            if (predicted_data[:7] == simulated_data[:7]).all():

                total_data_100[i] = np.append(simulated_data, predicted_data[7:10])
                # total_data_50.append(simulated_data + predicted_data[7:10])
                i += 1
    df_50 = pd.DataFrame({"tCu": total_data_100[:,0],"wCu": total_data_100[:,1],"tLam":total_data_100[:,2], "nLam": total_data_100[:,3], "aln": total_data_100[:,4],
                          "tsu": total_data_100[:,5], "freq": total_data_100[:,6], "real_Q":total_data_100[:,7], "real_R": total_data_100[:,8],"real_L": total_data_100[:,9],
                          "pre_Q":total_data_100[:,10], "pre_R":total_data_100[:,11], "pre_L":total_data_100[:,12]})
    df_50.to_csv(f"../{model_path}/Data_compare/compare_100.csv", index=False)
