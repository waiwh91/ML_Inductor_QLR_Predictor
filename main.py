import os
import pandas as pd
import data_processor
import trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

def process_data(path = "RLQ"):
    processor = data_processor.data_processor(path)

    processor.process_dir()


def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    data = pd.read_csv("data.csv").to_numpy()

    x_train, y_train, x_test, y_test = trainer.split_data(data, 0.333)

    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    f_train = x_train[:, 6]
    q_train = y_train[:, 0]
    r_train = y_train[:, 1]
    l_train = y_train[:, 2]

    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    f_test = x_test[:, 6]

    ############# 设定切分
    # dataset = TensorDataset(x_train[:,:6], f_train,y_train[:,1:3], q_train)
    dataset = TensorDataset(torch.log(x_train[:, :6]), torch.log(f_train), torch.log(y_train[:, 1:3]),
                            torch.log(q_train))
    batchsize = 64
    dataloader = DataLoader(dataset, batchsize, shuffle=False)

    ######开始训练

    model = trainer.PINN()
    model.to(device)
    trainer.train(model, dataloader, epoches=2500, alpha=1.0, beta=10.0)

    # trainer.test(model, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))

    mpe_q, mpe_r, mpe_l = trainer.test(model,
                 (torch.log(x_test[:, :6]), torch.log(x_test[:, 6]), torch.log(y_test[:, 0]), torch.log(y_test[:, 1]),
                  torch.log(y_test[:, 2])))

    torch.save(model.state_dict(), "models/PINN_model.pth")
    return mpe_q, mpe_r, mpe_l

def error_check():
    process_data()
    q = 0
    r = 0
    l = 0
    for i in range(10):
        mpe_q, mpe_r, mpe_l = train_model()
        q += mpe_q
        r += mpe_r
        l += mpe_l

    q /= 10
    r /= 10
    l /= 10

    print("MPE Q:", q*100, "%")
    print("MPE R:", r*100, "%")
    print("MPE L:", l*100, "%")


if __name__ == '__main__':

    # process_data()
    # train_model()

    error_check()