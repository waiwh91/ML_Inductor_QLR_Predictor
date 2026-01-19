from models.model_design import parameter_model
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_para():

    data = pd.read_csv("../inductor_w_tabfpn/output.csv").to_numpy()
    data = torch.from_numpy(data).float().to(device)

    test_data = pd.read_csv("training_csv/pinn_data.csv").to_numpy()
    test_data = torch.from_numpy(test_data).float().to(device)


    test_x = torch.cat([test_data[:, 1:6], test_data[:, 7:]], dim=1)
    test_y = test_data[:, 0]
    test_f = test_data[:, 6]

    print(test_x.shape, test_y.shape, test_f.shape)

    x = torch.log(torch.cat([data[:,1:6],data[:,7:]], dim=1))

    y = torch.log(data[:,0])

    f = data[:,6]


    dataset = TensorDataset(x, f, y)
    batchsize = 64
    dataloader = DataLoader(dataset, batchsize, shuffle=False)


    para_model = parameter_model.PINN()
    para_model.to(device)
    parameter_model.train(para_model, dataloader, epoches=400)

    pred_y = para_model(test_x, test_f)

    print(mean_absolute_percentage_error(pred_y.detach().cpu().numpy(), test_y.detach().cpu().numpy()))

    torch.save(para_model.state_dict(), "saved_models/tcu_model.pth")

if __name__ == '__main__':
    train_para()
