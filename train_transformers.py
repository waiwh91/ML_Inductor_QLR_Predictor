import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import data_process.ansys_integrator as data_processor
from models.model_design import transformers_model
from models import spliter
import models.dataloader
from models.model_design import pinn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pre_train_data_generator():


    x_pretrain = torch.rand(40000, 7).to(device)  # 输入维度6
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

    x_pretrain =  torch.log(x_pretrain)
    pinn_model = pinn.PINN()
    pinn_model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/saved_models/PINN_model.pth'))
    pinn_model.to(device)
    # pre_train_data = pre_train_data_generator()

    y_pred = pinn_model(x_pretrain[:,:6], x_pretrain[:,6])

    return x_pretrain, y_pred.detach()


def train_transformers():

    model = transformers_model.PINNTransformer()
    model.to(device)

    # ######### Pre train
    # x_pretrain, y_pretrain = pre_train_data_generator()
    # pre_train_dataloader = models.dataloader.transformers_dataloader(x_pretrain, y_pretrain, 2048)
    #
    #
    #
    # print("start Pre training")
    # transformers_model.train(model, pre_train_dataloader, epoches=300, alpha=1.0, beta=10)
    # print("Pre training done")
    #
    # torch.save(model.state_dict(), "saved_models/Pre_trained_PINNtransformers_model.pth")

    model.load_state_dict(torch.load('saved_models/Pre_trained_PINNtransformers_model.pth'))

    processor = data_processor.data_processor("RLQ/transformers_RLQ", "training_csv/transformers_data.csv")
    processor.process_dir()

    data = pd.read_csv("training_csv/transformers_data.csv").to_numpy()

    x_train, y_train, x_test, y_test = spliter.split_data(data, 1.0)
    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    dataloader = models.dataloader.transformers_dataloader(x_train,y_train,32)

    transformers_model.train(model, dataloader, epoches=300, alpha=1.0, beta=10)


     # pinn.test(models, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))
    torch.save(model.state_dict(), "saved_models/PINNtransformers_model.pth")

    # mpe_q, mpe_r, mpe_l = pinn.test(model,
    #                             (torch.log(x_test[:, :6]), torch.log(x_test[:, 6]), torch.log(y_test[:, 0]), torch.log(y_test[:, 1]),
    #                             torch.log(y_test[:, 2])))

if __name__ == "__main__":
    train_transformers()
