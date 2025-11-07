import torch
import numpy as np
import pandas as pd
from models.model_design import pinn_model
from models.model_design import transformers_model
from models.model_design import transformer_inter_model
from models.model_design import PINN_inter_model

def transformer_predict(x_train, model, f_train, output_csv = 'Parameter_impact/predicted_csv/predicted.csv'):
    r_pre, l_pre = model(torch.log(x_train)).T.detach().numpy()

    omega = np.log(2) + np.log(torch.pi) + torch.log(f_train)
    Q_pre = omega + l_pre - r_pre

    output_df = pd.DataFrame(
        {"tCu": x_train[:, 0], "wCu": x_train[:, 1], "tLam": x_train[:, 2], "nLam": x_train[:, 3],
         "aln": x_train[:, 4], "tsu": x_train[:, 5], "freq": f_train, "Pre_Q": np.exp(Q_pre),
         "Pre_R": np.exp(r_pre), "Pre_L": np.exp(l_pre)})

    output_df.to_csv(output_csv, index=False)

def pinn_predict(x_train, model, f_train, output_csv = 'Parameter_impact/predicted_csv/predicted.csv'):

    r_pre, l_pre = model(torch.log(x_train[:,:6]),torch.log(f_train)).T.detach().numpy()

    omega = np.log(2) + np.log(torch.pi) + torch.log(f_train)
    Q_pre = omega + l_pre - r_pre

    output_df = pd.DataFrame(
        {"tCu": x_train[:, 0], "wCu": x_train[:, 1], "tLam": x_train[:, 2], "nLam": x_train[:, 3],
         "aln": x_train[:, 4], "tsu": x_train[:, 5], "freq": f_train, "Pre_Q": np.exp(Q_pre),
         "Pre_R": np.exp(r_pre), "Pre_L": np.exp(l_pre)})

    output_df.to_csv(output_csv, index=False)



def predict(model, input_csv, output_csv):


    data = pd.read_csv(input_csv).to_numpy()

    # data = add_frequency.freq_adder(data, "Parameter_impact/generated_parameters/freq.csv")

    x_train = torch.from_numpy(data[:, :7]).int()
    f_train = torch.from_numpy(data[:, 6]).float()

    #### Determine which models type
    if isinstance(model, pinn_model.PINN):
        pinn_predict(x_train, model, f_train, output_csv)
    elif isinstance(model, transformers_model.PINNTransformer):
        transformer_predict(x_train, model, f_train, output_csv)
    elif isinstance(model, transformer_inter_model.PINNTransformer):
        transformer_predict(x_train, model, f_train, output_csv)
    elif isinstance(model, PINN_inter_model.PINN):
        pinn_predict(x_train, model, f_train, output_csv)
