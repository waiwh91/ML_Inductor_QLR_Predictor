import torch
import numpy as np
import pandas as pd
from model_design import transformers_model

model = transformers_model.PINNTransformer()
model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/Transformers/models/PINNtransformers_interpolation_model.pth'))





data = pd.read_csv("../data.csv").to_numpy()


x_train_100 = torch.from_numpy(data[:, :7]).int()
f_train_100 = torch.from_numpy(data[:, 6]).float()


# print(x_train)

# print(f_train)

########### 100% shift Data range
r_pre_100, l_pre_100 = model(torch.log(x_train_100)).T.detach().numpy()

omega = np.log(2) + np.log(torch.pi) + torch.log(f_train_100)
Q_pre_100 = omega + l_pre_100 - r_pre_100

output_df = pd.DataFrame({"tCu": x_train_100[:, 0], "wCu": x_train_100[:,1], "tLam":x_train_100[:,2], "nLam":x_train_100[:,3], "aln":x_train_100[:,4], "tsu":x_train_100[:,5], "freq":f_train_100, "Pre_Q": np.exp(Q_pre_100), "Pre_R": np.exp(r_pre_100), "Pre_L": np.exp(l_pre_100)})

output_df.to_csv("predicted_output.csv", index=False)

