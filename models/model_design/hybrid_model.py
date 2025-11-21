import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

######### Define PINN models 定义模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 8)
        # self.ln1 = nn.LayerNorm(8)

        self.fc2 = nn.Linear(8+1, 32)

        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16, 6)
        self.fc5 = nn.Linear(6, 2)

    def forward(self, x, extra):
        # h = torch.tanh(self.fc1(x))
        h = nn.functional.silu(self.fc1(x))
        # h = self.ln1(h)

        extra = extra.unsqueeze(1)
        h = torch.cat([h, extra], dim=1)
        # h = torch.tanh(self.fc2(h))
        h = nn.functional.silu(self.fc2(h))
        h = self.ln2(h)
        # h = torch.tanh(self.fc3(h))
        h = nn.functional.silu(self.fc3(h))
        h = nn.functional.silu(self.fc4(h))
        out = self.fc5(h)
        # out = nn.Softmax(dim=1)(out)
        return out


class ResidualTransformer(nn.Module):
    def __init__(self, input_dim = 10, nhead=2, d_model = 16, output_dim = 2,num_layers=1,dim_feedforward=32, dropout=0.2):
        super(ResidualTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(input_dim)
        # 输入线性映射到 d_model
        self.input_fc = nn.Linear(1, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        # Positional encoding (可选，这里直接用 learnable embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, 11, d_model))

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

        h = h + self.pos_embedding[:, :h.size(1), :]  # add positional embedding
        h = self.transformer_encoder(h)  # [batch, seq_len=1, d_model]
        # h = h.squeeze(-1)  # remove seq_len dim
        h = h[:, -1, :]
        # print(h.shape)
        out = self.output_fc(h)

        return out


class GMM(nn.Module):
    def __init__(self, n_components=3):
        super().__init__()
        self.n_components = n_components
        # 初始化均值、方差、权重
        self.means = nn.Parameter(torch.randn(n_components, 1))
        self.log_vars = nn.Parameter(torch.zeros(n_components, 1))
        self.log_weights = nn.Parameter(torch.zeros(n_components))

    def forward(self, x):
        # x: [N,1]
        x = x.unsqueeze(1)  # [N,1] -> [N,1,1]
        means = self.means.unsqueeze(0)  # [1,K,1]
        vars = torch.exp(self.log_vars).unsqueeze(0)  # [1,K,1]
        log_weights = torch.log_softmax(self.log_weights, dim=0)  # [K]

        # Gaussian log likelihood
        log_prob = -0.5 * ((x - means) ** 2 / vars + torch.log(2 * torch.pi * vars))  # [N,K,1]
        log_prob_sum = log_prob.sum(dim=2)  # [N,K]
        log_prob = log_prob_sum + log_weights  # [N,K], 广播成功
        return log_prob


def train_pinn(model, dataloader, epoches = 2000, alpha = 1.0, beta = 6.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.MSELoss()
    loss_q = nn.SmoothL1Loss()


    for epoch in range(epoches):
        epoch_loss = 0.0

        for batch_x, batch_f, batch_y, batch_q in dataloader:

            preds = model(batch_x, batch_f)

            loss_data = loss_fn(preds, batch_y)

            # R_pre = preds.detach().cpu().numpy()[:,0]
            # L_pre = preds.detach().cpu().numpy()[:,1]
            #
            #
            # f = batch_f.detach().cpu().numpy()

            R_pre = preds.detach()[:, 0]
            L_pre = preds.detach()[:, 1]

            f = batch_f.detach()


            omega = np.log(2) + np.log(torch.pi) + f
            Q_pre = omega + L_pre - R_pre


            # for i in range(len(Q_pre)):
            #     if Q_pre[i] > np.log(1000):
            #         Q_pre[i] = Q_pre[i] - np.log(1000)


            # loss_physics = loss_fn(torch.from_numpy(Q_pre).to(device), batch_q)
            loss_physics = loss_fn(Q_pre, batch_q)



            loss = alpha * loss_data + beta * loss_physics             ############ α 和 β 用来指定两种损失函数哪个比重大

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epoches}, Loss = {epoch_loss/len(dataloader):.6f}")

def train_GMM(model, residual, epoches = 600):

    optimizer_gmm = optim.Adam(model.parameters(), lr=1e-2)

    # EM-like训练
    for epoch in range(epoches):
        log_prob = model(residual)
        # 负对数似然
        loss = -torch.logsumexp(log_prob, dim=1).mean()
        optimizer_gmm.zero_grad()
        loss.backward()
        optimizer_gmm.step()
        if (epoch + 1) % 100 == 0:
            print(f"GMM Epoch {epoch + 1}, Loss: {loss.item():.4f}")



def train_residualTrans(model, x_train, residual, epoches = 300):

    optimizer_tr = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # 用残差训练Transformer
    for epoch in range(epoches):
        optimizer_tr.zero_grad()
        res_pred = model(x_train)

        loss = loss_fn(res_pred, residual)
        loss.backward()
        optimizer_tr.step()
        if (epoch + 1) % 100 == 0:
            print(f"Transformer Epoch {epoch + 1}, Loss: {loss.item():.4f}")