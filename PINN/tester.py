import torch
import numpy as np
import pandas as pd

model = trainer.PINN()
model.load_state_dict(torch.load('saved_models/PINN_model.pth'))

data = pd.read_csv("simulation_test/simulation_data.csv").to_numpy()

x_train = torch.from_numpy(data[:, :6]).int()
f_train = torch.from_numpy(data[:, 6]).float()
# print(x_train)

# print(f_train)


r_pre, l_pre = model(torch.log(x_train), torch.log(f_train)).T.detach().numpy()

omega = np.log(2) + np.log(torch.pi) + torch.log(f_train)
Q_pre = omega + l_pre - r_pre

output_df = pd.DataFrame({"tCu": x_train[:, 0], "wCu": x_train[:,1], "tLam":x_train[:,2], "nLam":x_train[:,3], "aln":x_train[:,4], "tsu":x_train[:,5], "freq":f_train,"Pre_R": np.exp(r_pre), "Pre_L": np.exp(l_pre),"Pre_Q": np.exp(Q_pre)})

output_df.to_csv("simulation_test/predicted_output.csv", index=False)
