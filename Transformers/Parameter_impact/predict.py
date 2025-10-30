import torch
import numpy as np
import pandas as pd
from model_design import transformers_model
import data_process.add_frequency as add_frequency

model = transformers_model.PINNTransformer()
model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/Transformers/models/PINNtransformers_model.pth'))




data_50 = pd.read_csv("generated_parameters/test_parameter_50.csv").to_numpy()
data_100 = pd.read_csv("generated_parameters/test_parameter_100.csv").to_numpy()

data_50 = add_frequency.freq_adder(data_50, "generated_parameters/freq_50.csv")
data_100 = add_frequency.freq_adder(data_100, "generated_parameters/freq_100.csv")

x_train_50 = torch.from_numpy(data_50[:, :7]).int()
f_train_50 = torch.from_numpy(data_50[:, 6]).float()

x_train_100 = torch.from_numpy(data_100[:, :7]).int()
f_train_100 = torch.from_numpy(data_100[:, 6]).float()


# print(x_train)

# print(f_train)

############ 50 % shift Data range

r_pre_50, l_pre_50 = model(torch.log(x_train_50)).T.detach().numpy()

omega = np.log(2) + np.log(torch.pi) + torch.log(f_train_50)
Q_pre_50 = omega + l_pre_50 - r_pre_50

output_df = pd.DataFrame({"tCu": x_train_50[:, 0], "wCu": x_train_50[:,1], "tLam":x_train_50[:,2], "nLam":x_train_50[:,3], "aln":x_train_50[:,4], "tsu":x_train_50[:,5], "freq":f_train_50,"Pre_Q": np.exp(Q_pre_50), "Pre_R": np.exp(r_pre_50), "Pre_L": np.exp(l_pre_50)})

output_df.to_csv("predicted_csv/predicted_50.csv", index=False)

########### 100% shift Data range
r_pre_100, l_pre_100 = model(torch.log(x_train_100)).T.detach().numpy()

omega = np.log(2) + np.log(torch.pi) + torch.log(f_train_100)
Q_pre_100 = omega + l_pre_100 - r_pre_100

output_df = pd.DataFrame({"tCu": x_train_100[:, 0], "wCu": x_train_100[:,1], "tLam":x_train_100[:,2], "nLam":x_train_100[:,3], "aln":x_train_100[:,4], "tsu":x_train_100[:,5], "freq":f_train_100, "Pre_Q": np.exp(Q_pre_100), "Pre_R": np.exp(r_pre_100), "Pre_L": np.exp(l_pre_100)})

output_df.to_csv("predicted_csv/predicted_100.csv", index=False)

