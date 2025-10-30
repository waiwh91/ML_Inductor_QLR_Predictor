import pandas as pd
from torch import device

import data_process.ansys_integrator
from model import spliter
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.model_design import pinn, transformers_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_data(path = "/home/martin/ML_Inductor_QLR_Predictor/Transformers/transformers_RLQ"):
    processor = data_process.ansys_integrator.data_processor(file_path=path)

    processor.process_dir()

def process_aln(path =  "/home/martin/ML_Inductor_QLR_Predictor/Transformers/aln"):
    processor = data_process.ansys_integrator.data_processor(file_path=path, output_path="aln.csv")
    processor.process_dir()


def pre_train_data_generator():


    x_pretrain = torch.rand(20000, 7).to(device)  # 输入维度6
    x_pretrain[:, 0] = 5 + 35 * x_pretrain[:, 0]
    x_pretrain[:, 1] = 50 + 200 * x_pretrain[:, 1]
    x_pretrain[:, 2] = 50 + 650 * x_pretrain[:, 2]
    x_pretrain[:, 3] = 4 + 28 * x_pretrain[:, 3]
    x_pretrain[:, 4] = 10 + 30 * x_pretrain[:, 4]
    x_pretrain[:, 5] = 2 + 12 * x_pretrain[:, 5]
    pre_train_f_tensor_list = torch.tensor([1, 25.75, 50.5, 75.25, 100]).to(device)
    for i in range(len(x_pretrain[:, 6])):
        idx = torch.randint(0, len(pre_train_f_tensor_list), (1,))
        x_pretrain[i, 6] = pre_train_f_tensor_list[idx]

    return torch.log(x_pretrain)

def pre_train_data_loader(x_pre_train,y_pred):

    R = y_pred[:,0].detach()
    L = y_pred[:,1].detach()
    f = x_pre_train[:,6]
    omega = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(torch.pi)) + f
    Q_pre = omega + L - R

    dataset = TensorDataset(x_pre_train,x_pre_train[:,6],y_pred, Q_pre.to(device))
    batchsize = 1024
    dataloader = DataLoader(dataset, batchsize, shuffle=False)
    return dataloader

def aln_train_dataloader():
    data = pd.read_csv("aln.csv").to_numpy()
    x_train = torch.from_numpy(data[:,:7]).float().to(device)
    y_train = torch.from_numpy(data[:,7:]).float().to(device)
    f_train = torch.from_numpy(data[:,6]).float().to(device)
    q_train = torch.from_numpy(data[:,7]).float().to(device)
    dataset = TensorDataset(torch.log(x_train[:, :7]), torch.log(f_train), torch.log(y_train[:, 1:3]),
                            torch.log(q_train))

    batchsize = 1
    dataloader = DataLoader(dataset, batchsize, shuffle=False)
    return dataloader

def train_model():

    print("device: ", device)
    data = pd.read_csv("data.csv").to_numpy()

    x_train, y_train, x_test, y_test = spliter.split_data(data, 1.0)

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
    dataset = TensorDataset(torch.log(x_train[:, :7]), torch.log(f_train), torch.log(y_train[:, 1:3]),
                            torch.log(q_train))
    batchsize = 32
    dataloader = DataLoader(dataset, batchsize, shuffle=False)

    ######开始训练
    ##########先预训练
    pinn_model = pinn.PINN()
    pinn_model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/PINN/saved_models/PINN_model.pth'))
    pinn_model.to(device)
    pre_train_data = pre_train_data_generator()

    y_pred = pinn_model(pre_train_data[:,:6], pre_train_data[:,6])
    print(y_pred.shape)

    pre_dataloader = pre_train_data_loader(pre_train_data,y_pred.detach())

    model = transformers_model.PINNTransformer()
    model.to(device)

    print("start Pre training")
    transformers_model.train(model, pre_dataloader, epoches=300, alpha=1.0, beta=10)
    print("Pre training done")
    print("start training")

    transformers_model.train(model, dataloader, epoches=1000, alpha=1.0, beta=50.0)
    print("training done")
    # print("start train aln")
    #
    # aln_dataloader = aln_train_dataloader()
    # transformers_model.train(model, aln_dataloader, epoches=300, alpha=1.0, beta=50.0)
    #
    # print("aln training done")
    torch.save(model.state_dict(), "../saved_models/PINNtransformers_model.pth")



    # trainer.test(model, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))

    mpe_q, mpe_r, mpe_l = transformers_model.test(model,
                                                  (torch.log(x_test[:, :7]), torch.log(x_test[:, 6]), torch.log(y_test[:, 0]), torch.log(y_test[:, 1]),
                  torch.log(y_test[:, 2])))


    return mpe_q, mpe_r, mpe_l

if __name__ == '__main__':

    process_data()
    process_aln()
    train_model()

    # error_check()