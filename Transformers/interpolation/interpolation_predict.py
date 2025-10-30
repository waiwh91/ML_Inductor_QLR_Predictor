import torch
import pandas as pd
from model.model_design import transformers_model
import predict

model = transformers_model.PINNTransformer()
model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/Transformers/saved_models/PINNtransformers_interpolation_model.pth'))

data = pd.read_csv("../data.csv").to_numpy()

predict.predict(model, data, "predicted_output.csv")
