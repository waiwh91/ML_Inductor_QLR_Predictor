from models.model_design import hybrid_model
import pandas as pd
import numpy as np
import torch

def hybrid_predict(pinn_model, trans_model,input_csv, output_csv = 'Parameter_impact/predicted_csv/predicted.csv'):
    data = pd.read_csv(input_csv).to_numpy()


    x_train = torch.from_numpy(data[:, :7]).int()
    f_train = torch.from_numpy(data[:, 6]).float()

    r_pre, l_pre = pinn_model(torch.log(x_train[:, :6]), torch.log(f_train)).T.detach().numpy()


    error_r, error_l = trans_model(torch.log(x_train)).T.detach().numpy()

    r_pre, l_pre = r_pre + error_r, l_pre + error_l


    omega = np.log(2) + np.log(torch.pi) + torch.log(f_train)


    Q_pre = omega + l_pre - r_pre

    output_df = pd.DataFrame(
        {"tCu": x_train[:, 0], "wCu": x_train[:, 1], "tLam": x_train[:, 2], "nLam": x_train[:, 3],
             "aln": x_train[:, 4], "tsu": x_train[:, 5], "freq": f_train, "Pre_Q": np.exp(Q_pre),
             "Pre_R": np.exp(r_pre), "Pre_L": np.exp(l_pre)})

    output_df.to_csv(output_csv, index=False)