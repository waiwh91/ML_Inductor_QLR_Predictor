import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


######### 划分数据集 split dataset into train/test

def split_data(data):
    x = data[:,:7]
    y = data[:,7:]

    x_train = x[:int(data.shape[0]*0.7), :]
    x_test = x[int(data.shape[0]*0.7):, :]

    y_train = y[:int(data.shape[0]*0.7), :]
    y_test = y[int(data.shape[0]*0.7):, :]

    return x_train, y_train, x_test, y_test

######### Define PINN model 定义模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Softplus()
        )

    def forward(self, x):
        # print(x.shape)
        return self.net(x)

def train(model, dataloader, epoches = 2000, alpha = 1.0, beta = 1.0):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    loss_q = nn.SmoothL1Loss()


    for epoch in range(epoches):
        epoch_loss = 0.0

        for batch_x, batch_y, batch_q in dataloader:

            preds = model(batch_x)

            loss_data = loss_fn(preds, batch_y)

            R_pre = preds.detach().numpy()[:,0]
            L_pre = preds.detach().numpy()[:,1]


            f = np.expm1(batch_x[:, 6].detach().numpy())
            omega = 2 * torch.pi * f
            Q_pre = omega * R_pre / L_pre

            loss_physics = loss_q(torch.log(Q_pre), torch.log(batch_q))

            loss = alpha * loss_data + beta * loss_physics             ############ α 和 β 用来指定两种损失函数哪个比重大

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epoches}, Loss = {epoch_loss/len(dataloader):.6f}")

def test(model, data):
    x, f, Q_data, R_data, L_data = data

    R_pre, L_pre = model(x).T.detach().numpy()

    R_pre = np.expm1(R_pre)
    L_pre = np.expm1(L_pre)

    omega = 2 * torch.pi * f
    Q_pre = omega * R_pre / L_pre

    dataframe = pd.DataFrame({"Rpre":R_pre.detach().numpy(), "Lpre":L_pre.detach().numpy(), "Qpre":Q_pre.detach().numpy(), "Rdata":R_data.detach().numpy(), "Ldata":L_data.detach().numpy(), "Qdata":Q_data.detach().numpy(), "f":f})
    dataframe.to_csv("output.csv", index=False)

    error_q_total = 0
    error_r_total = 0
    error_l_total = 0

    PREQ = Q_pre.detach().numpy()
    DATQ = Q_data.detach().numpy()

    PRER = R_pre.detach().numpy()
    DATR = R_data.detach().numpy()

    PREL = L_pre.detach().numpy()
    DATL = L_data.detach().numpy()

    DATF = f.detach().numpy()
    sumf = 0
    for i in range(len(PREQ)):

        if DATF[i] == 1:
            sumf += 1

            error_q_total = error_q_total + (abs(PREQ[i] - DATQ[i]).item() / DATQ[i])
            error_r_total = error_r_total + (abs(PRER[i] - DATR[i]).item() / DATR[i])
            error_l_total = error_l_total + (abs(PREL[i] - DATL[i]).item() / DATL[i])


    print(f"RMSE of Q: {(error_q_total/sumf)}")
    print(f"RMSE of R: {(error_r_total/sumf)}")
    print(f"RMSE of L: {(error_l_total/sumf)}")

