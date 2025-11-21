import pandas as pd
import numpy as np
import torch
from sympy.codegen import Print
from torch.utils.data import DataLoader, TensorDataset
import data_process.ansys_integrator as data_processor
from models.model_design import hybrid_model
from models import spliter
import models.dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_hybrid():
    processor = data_processor.data_processor("RLQ/interpolation_RLQ", "training_csv/interpolation_data.csv")
    processor.process_dir()

    data = pd.read_csv("training_csv/interpolation_data.csv").to_numpy()

    x_train, y_train, x_test, y_test = spliter.split_data(data, 1.0)
    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    dataloader = models.dataloader.pinn_dataloader(x_train, y_train, 6)

    model = hybrid_model.PINN()
    model.to(device)
    hybrid_model.train_pinn(model, dataloader, epoches=1500, alpha=1.0, beta=10.0)

    # pinn.test(models, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))
    torch.save(model.state_dict(), "saved_models/PINN_inter_model.pth")


    y_pinn = model(x_train[:,:6], x_train[:,6]).detach()

    y_pinn = torch.tensor(y_pinn.detach())


    residual = torch.log(torch.tensor(y_train[:,1:3])) - y_pinn



    trans_model = hybrid_model.ResidualTransformer()
    trans_model.to(device)
    hybrid_model.train_residualTrans(trans_model, x_train, residual, epoches = 500)


    torch.save(trans_model.state_dict(), "saved_models/hybrid_model/residual_trans_model.pth")


if __name__ == "__main__":
    train_hybrid()