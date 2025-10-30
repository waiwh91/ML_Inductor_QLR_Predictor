import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import data_process.ansys_integrator as data_processor
from models.model_design import pinn
from models import spliter
from models import dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_pinn():

    processor = data_processor.data_processor("RLQ/pinn_RLQ", "training_csv/pinn_data.csv")
    processor.process_dir()

    data = pd.read_csv("training_csv/pinn_data.csv").to_numpy()

    x_train, y_train, x_test, y_test = torch.from_numpy(spliter.split_data(data, 0.7)).float().to(device)
    dataloader = dataloader.dataloader(x_train,y_train,16)

    model = pinn.PINN()
    model.to(device)
    pinn.train(model, dataloader, epoches=1200, alpha=1.0, beta=10.0)

     # pinn.test(models, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))
    torch.save(model.state_dict(), "../saved_models/PINN_model.pth")

    mpe_q, mpe_r, mpe_l = pinn.test(model,
                                (torch.log(x_test[:, :6]), torch.log(x_test[:, 6]), torch.log(y_test[:, 0]), torch.log(y_test[:, 1]),
                                torch.log(y_test[:, 2])))

if __name__ == "__main__":
    train_pinn()