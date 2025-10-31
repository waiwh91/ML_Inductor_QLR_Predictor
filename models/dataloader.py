import pandas as pd
from models.model_design import interpolation_model
import models.spliter as spliter
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transformers_dataloader(x_train, y_train, batchsize):
    # R = y_train[:,0].detach()
    # L = y_train[:,1].detach()
    # f = x_train[:,6]
    # omega = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(torch.pi)) + f
    # Q_pre = omega + L - R

    dataset = TensorDataset(torch.log(x_train),torch.log(x_train[:,6]),torch.log(y_train[:,1:3]), torch.log(y_train[:,0]))
    batchsize = batchsize
    dataloader = DataLoader(dataset, batchsize, shuffle=False)
    return dataloader

def pinn_dataloader(x_train, y_train, batchsize):
    # R = y_train[:,0].detach()
    # L = y_train[:,1].detach()
    # f = x_train[:,6]
    # omega = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(torch.pi)) + f
    # Q_pre = omega + L - R

    dataset = TensorDataset(torch.log(x_train[:,:6]),torch.log(x_train[:,6]),torch.log(y_train[:,1:3]), torch.log(y_train[:,0]))
    batchsize = batchsize
    dataloader = DataLoader(dataset, batchsize, shuffle=False)
    return dataloader
