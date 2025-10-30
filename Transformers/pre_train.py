import torch
import torch.optim as optim


def train(model, dataloader, epoches = 200,alpha = 1.0, beta = 10.0):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # 使用PINN预训练

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pinn_model = PINN.trainer.PINN()
    pinn_model.load_state_dict(torch.load('/home/martin/ML_Inductor_QLR_Predictor/PINN/models/PINN_model.pth'))
    pinn_model.to(device)

    x_pretrain = torch.rand(100000, 7).to(device)  # 输入维度6
    x_pretrain[:,0] = 5 + 35 * x_pretrain[:,0]
    x_pretrain[:,1] = 50 + 200 * x_pretrain[:,1]
    x_pretrain[:,2] = 50 + 650 * x_pretrain[:,2]
    x_pretrain[:,3] = 4 + 28 * x_pretrain[:,3]
    x_pretrain[:,4] = 10 + 30 * x_pretrain[:,4]
    x_pretrain[:,5] = 2 + 12 * x_pretrain[:,5]
    pre_train_f_tensor_list = torch.tensor([1, 25.75, 50.5, 75.25, 100]).to(device)
    for i in range(len(x_pretrain[:,6])):
        idx = torch.randint(0, len(pre_train_f_tensor_list),(1,))
        x_pretrain[i,6] = pre_train_f_tensor_list[idx]

    y_pinn = pinn_model(x_pretrain[:,:6],x_pretrain[:,6]).detach()
    for epoch in range(600):
        epoch_loss = 0
        y_pred = model(x_pretrain)
        loss = torch.mean((y_pred - y_pinn) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        if epoch % 100 == 0:
            print(f"Pre_train_Epoch {epoch + 1}/{epoches}, pre_train_Loss = {epoch_loss / len(dataloader):.6f}")
    # torch.save(model.state_dict(), "models/transformer_pretrained_from_pinn.pt")

    for epoch in range(epoches):
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