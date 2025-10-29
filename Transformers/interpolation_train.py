import pandas as pd
import PINN.data_processor
import transformers_model
import spliter
import torch
from torch.utils.data import DataLoader, TensorDataset




def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    data = pd.read_csv("Parameter_impact/simulator_output_data/csv_100.csv").to_numpy()

    x_train, y_train, x_test, y_test = spliter.split_data(data, 0.9)

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
    batchsize = 16
    dataloader = DataLoader(dataset, batchsize, shuffle=False)

    ######开始训练

    model = transformers_model.PINNTransformer()
    model.to(device)
    transformers_model.train(model, dataloader, epoches=800, alpha=1.0, beta=50.0)

    # trainer.test(model, (x_test[:,:6], x_test[:,6], y_test[:,0], y_test[:,1], y_test[:,2]))

    mpe_q, mpe_r, mpe_l = transformers_model.test(model,
                                       (torch.log(x_test[:, :7]), torch.log(x_test[:, 6]), torch.log(y_test[:, 0]), torch.log(y_test[:, 1]),
                  torch.log(y_test[:, 2])))

    torch.save(model.state_dict(), "models/PINNtransformers_model.pth")
    return mpe_q, mpe_r, mpe_l

if __name__ == '__main__':

    # process_data()
    train_model()

    # error_check()