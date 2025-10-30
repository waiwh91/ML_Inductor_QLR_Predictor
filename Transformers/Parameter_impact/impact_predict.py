import torch
import pandas as pd
from model.model_design import transformers_model
import data_process.add_frequency as add_frequency
import predict

model = transformers_model.PINNTransformer()
model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/Transformers/saved_models/PINNtransformers_model.pth'))




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

# 50% Predict
predict.predict(model, data_50, "predicted_csv/predicted_50.csv")


########### 100% shift Data range

predict.predict(model, data_100, "predicted_csv/predicted_100.csv")
