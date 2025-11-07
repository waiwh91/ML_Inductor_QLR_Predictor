import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


class PINNTransformer(nn.Module):
    def __init__(self, input_dim=7, output_dim=2, d_model=16, nhead=1, num_layers=3, dim_feedforward=32, dropout=0.1):
        super(PINNTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(input_dim)
        # 输入线性映射到 d_model
        self.input_fc = nn.Linear(1, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        # Positional encoding (可选，这里直接用 learnable embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, 8, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出映射
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        """
        # [batch, input_dim] -> [batch, seq_len=1, d_model]
        h = x.unsqueeze(-1)
        # h = self.input_norm(h)

        h = self.input_fc(h)
        # print(h.shape)
        h = self.layer_norm_in(h)

        # print(self.pos_embedding.shape)

        h = h + self.pos_embedding[:, h.size(1), :]  # add positional embedding
        h = self.transformer_encoder(h)  # [batch, seq_len=1, d_model]
        # h = h.squeeze(-1)  # remove seq_len dim
        h = h[:, -1, :]
        # print(h.shape)
        out = self.output_fc(h)

        return out


def physics_informed_loss_function(y_pred, y_true, batch_f,batch_q, alpha, beta):


    loss_fn = nn.MSELoss()
    loss_data = loss_fn(y_pred, y_true)

    R_pre = y_pred.detach()[:, 0]
    L_pre = y_pred.detach()[:, 1]
    f = batch_f.detach()
    omega = torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(torch.pi)) + f
    Q_pre = omega + L_pre - R_pre
    loss_physics = loss_fn(Q_pre, batch_q)
    loss = alpha * loss_data + beta * loss_physics

    return loss

def train(model, dataloader, epoches = 200,alpha = 1.0, beta = 10.0):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epoches+1):
        epoch_loss = 0

        for batch_x,  batch_f,batch_y, batch_q in dataloader:
            preds = model(batch_x)
            loss = physics_informed_loss_function(preds, batch_y, batch_f,batch_q, alpha, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{epoches}, Loss = {epoch_loss / len(dataloader):.6f}")


def test(model, data):
    x, f, Q_data, R_data, L_data = data



    R_pre, L_pre = model(x).T.detach().cpu().numpy()

    omega = np.log(2) + np.log(torch.pi) + f
    Q_pre = omega.cpu().numpy() + L_pre - R_pre

    # for i in range(len(Q_pre)):
    #     if Q_pre[i] > np.log(1000):
    #         Q_pre[i] = Q_pre[i] - np.log(1000)
    R_data = R_data.detach().cpu()
    L_data = L_data.detach().cpu()
    Q_data = Q_data.detach().cpu()
    f = f.detach().cpu()

    dataframe = pd.DataFrame({"Rpre":np.exp(R_pre), "Lpre":np.exp(L_pre), "Qpre":np.exp(Q_pre), "Rdata":np.exp(R_data), "Ldata":np.exp(L_data), "Qdata":np.exp(Q_data), "f":np.exp(f)})
    dataframe.to_csv("training_csv/interpolation_output.csv", index=False)

    error_q_total = 0
    error_r_total = 0
    error_l_total = 0

    PREQ = Q_pre
    DATQ = Q_data.detach().cpu().numpy()

    PRER = R_pre
    DATR = R_data.detach().cpu().numpy()

    PREL = L_pre
    DATL = L_data.detach().cpu().numpy()

    DATF = f.detach().numpy()
    sumf = 0
    for i in range(len(PREQ)):


        sumf += 1

        error_q_total = error_q_total + (abs(PREQ[i] - DATQ[i]).item() / DATQ[i])
        error_r_total = error_r_total + (abs(PRER[i] - DATR[i]).item() / DATR[i])
        error_l_total = error_l_total + (abs(PREL[i] - DATL[i]).item() / DATL[i])


    print(f"RMSE of Q: {(error_q_total/sumf)}")
    print(f"RMSE of R: {(error_r_total/sumf)}")
    print(f"RMSE of L: {(error_l_total/sumf)}")

    return error_q_total/sumf, error_r_total/sumf, error_l_total/sumf

