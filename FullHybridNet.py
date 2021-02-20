import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from data_loader import SinglePredDatasetTrain
from torch.utils.data import DataLoader
from End2EndNet import TConvBlock
import torchsummary

PATH = './FH_v3.pth'

class AccelErrorNet(nn.Module):
    # Deep Neural Network for motor thrust prediction
    def __init__(self, lookback, pred_steps):
        super(AccelErrorNet, self).__init__()
        L = lookback
        P = pred_steps
        K = 8
        d = 2
        t = 28
        self.L = L
        self.P = P
        self.t = t
        self.tconv1 = TConvBlock(L + P, 16, 16, K, d)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(L + P, 16, 16, K, d)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(L + P, 16, 32, K, d)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(t, 32, 16, K, d)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.relu4 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(t * 16, 128)
        self.relu5 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 6)

    def forward(self, input):
        # Assume X: batch by length by channel size
        # print(input.shape)
        x = self.relu1(self.bn1(self.tconv1(input)))
        x = self.relu2(self.bn2(self.tconv2(x)))
        x = self.relu3(self.bn3(self.tconv3(x)))
        x = self.relu4(self.bn4(self.tconv4(x[:, :, (self.L + self.P - self.t):])))
        x = torch.flatten(x, 1, 2)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x


class MotorHybrid(nn.Module):
    # Deep Neural Network for motor thrust prediction
    def __init__(self, lookback, pred_steps):
        super(MotorHybrid, self).__init__()
        L = lookback
        P = pred_steps
        K = 8
        d = 2
        t = 28
        self.L = L
        self.P = P
        self.t = t
        self.tconv1 = TConvBlock(L + P, 16, 16, K, d)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(L + P, 16, 16, K, d)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(L + P, 16, 32, K, d)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(t, 32, 16, K, d)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.relu4 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(t * 16, 128)
        self.relu5 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 4)

    def forward(self, input):
        # Assume X: batch by length by channel size
        # print(input.shape)
        x = self.relu1(self.bn1(self.tconv1(input)))
        x = self.relu2(self.bn2(self.tconv2(x)))
        x = self.relu3(self.bn3(self.tconv3(x)))
        x = self.relu4(self.bn4(self.tconv4(x[:, :, (self.L + self.P - self.t):])))
        x = torch.flatten(x, 1, 2)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x


class QuadrotorDynamics(nn.Module):
    def __init__(self, l, m, d, kt, kr, ixx, iyy, izz, lookback, pred_steps):
        super().__init__()
        self.l = l
        self.m = m
        self.d = d
        self.kt = kt
        self.kr = kr
        self.I = torch.tensor([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
        self.accel_net = AccelErrorNet(lookback, pred_steps)
        self.motor_net = MotorHybrid(lookback, pred_steps)
        # torchsummary.summary(self.accel_net, (16, 65))
        # torchsummary.summary(self.motor_net, (16, 65))
        self.torque_mat = torch.tensor([[1, 1, 1, 1],
                          [0.707 * self.l, -0.707 * self.l, -0.707 * self.l, 0.707 * self.l],
                          [-0.707 * self.l, -0.707 * self.l, 0.707 * self.l, 0.707 * self.l],
                          [-self.d, self.d, -self.d, self.d]])
        self.select = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1/self.m, 0, 0, 0]]).type(torch.FloatTensor)
        self.g = torch.tensor([[0], [0], [9.8067]])

    def forward(self, t, input):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        state = torch.transpose(input[:, :, -2], 0, 1)
        ang = state[0:3, 0]
        rate = state[6:9, :]
        vel = state[9:12, :]

        # update torques
        thrusts = torch.transpose(self.motor_net(input), 0, 1)
        torques = torch.mm(self.torque_mat, thrusts)

        # Calculate rotation matrix
        s_phi = (torch.sin(ang[0])).item()
        c_phi = (torch.cos(ang[0])).item()
        s_theta = (torch.sin(ang[1])).item()
        c_theta = (torch.cos(ang[1])).item()
        s_psi = (torch.sin(ang[2])).item()
        c_psi = (torch.cos(ang[2])).item()

        rbi = torch.tensor(
            [[c_theta * c_psi, c_psi * s_theta * s_phi - c_phi * s_psi, c_phi * c_psi * s_theta + s_phi * s_psi],
             [c_theta * s_psi, s_psi * s_theta * s_phi + c_phi * c_psi, c_phi * s_psi * s_theta - s_phi * c_psi],
             [-s_theta, c_theta * s_phi, c_theta * c_phi]])

        M = torch.tensor([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])

        vel_dot = torch.mm(rbi, torch.mm(self.select, torques)) - self.kt * vel - self.g
        m_inv = torch.inverse(M)
        ang_dot = torch.mm(m_inv, rate)
        rate_dot = torch.mm(torch.inverse(self.I), torques[1:] - torch.cross(rate, torch.mm(self.I, rate), dim=0)
                             - self.kr * rate)

        state_dot = torch.transpose(torch.cat([ang_dot, vel, rate_dot, vel_dot, torch.zeros((4, 1))]), 0, 1)
        state_dot[:, 6:12] += self.accel_net(input)
        output = torch.zeros(input.shape)
        output[:, :, -2] = state_dot
        return output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    l = 0.211  # length (m)
    d = 1.7e-5  # blade parameter
    m = 1  # mass (kg)
    kt = 2.35e-14  # translational drag coefficient
    kr = 0.0099  # rotational drag coefficient
    ixx = 0.002  # moment of inertia about X-axis
    iyy = 0.002  # moment of inertia about Y-axis
    izz = 0.001  # moment of inertia about Z-axis

    bs = 1
    lookback = 64
    lr = 0.001
    wd = 0.0005
    epochs = 30
    pred_steps = 1

    tv_set = SinglePredDatasetTrain('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    func = QuadrotorDynamics(l, m, d, kt, kr, ixx, iyy, izz, lookback, pred_steps)
    func.to(device)
    optimizer = optim.Adam(list(func.parameters()), lr=lr)
    loss_f = nn.MSELoss()
    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0
    print("Training Length: {}".format(int(train_len / bs)))

    for epoch in range(1, epochs + 1):
        # Training
        print("Training")
        func.train(True)
        epoch_train_losses = []
        epoch_train_loss = 0
        moving_av = 0
        epoch_val_losses = []
        i = 0

        for data in train_loader:
            optimizer.zero_grad()
            raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2).to(device)  # Load Input data
            # n_raw = raw_input.numpy()
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2).to(device)  # Load labels
            # n_label = label.numpy()
            output_gt = label[0, 6:12, 0]
            feedforward = torch.zeros(label.shape)
            feedforward[:, -4:, :] = label[:, -4:, :]
            input = torch.cat((raw_input, feedforward), 2)
            # n_input = input.numpy()
            output = odeint(func, input, torch.tensor([0, 0.01]))
            pred = output[1, 0, 6:12, -2]
            loss = loss_f(pred, output_gt)
            epoch_train_loss += loss.item() / train_len
            moving_av += loss.item()
            if loss.requires_grad:
                loss.backward()
            optimizer.step()
            if loss.item() > 10:
                print("edgecase")

            i += 1
            if i % 50 == 0:
                print("Training {}% finished".format(round(100 * i / train_len, 4)))
                print(moving_av / 50)
                moving_av = 0

        train_loss.append(epoch_train_loss)
        print("Training Error for this Epoch: {}".format(epoch_train_loss))

        print("Validation")
        func.train(False)
        func.eval()
        epoch_val_loss = 0
        i = 0
        with torch.no_grad():
            for data in val_loader:
                raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
                label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels
                output_gt = label[0, 6:12, 0]
                feedforward = torch.zeros(label.shape)
                feedforward[:, -4:, :] = label[:, -4:, :]
                input = torch.cat((raw_input, feedforward), 2)

                optimizer.zero_grad()  # Reset gradients
                output = odeint(func, input, torch.tensor([0, 0.01]))
                loss = loss_f(output[1, 0, 6:12, -2], output_gt)
                epoch_val_losses.append(loss)

                i += 1
                if i % 100 == 0:
                    print(i)

            val_loss.append(np.mean(epoch_val_losses))
            print("Validation Error for this Epoch: {}".format(np.mean(epoch_val_losses)))
            if best_epoch == 0 or np.mean(epoch_val_losses) < best_loss:
                best_loss = np.mean(epoch_val_losses)
                best_epoch = epoch
                torch.save(func.state_dict(), PATH)

            # Plotting
            plt.plot(train_loss, linewidth=2)
            plt.plot(val_loss, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend(["Training Loss", "Validation Loss"])
            plt.savefig("FH_v3_train_intermediate.png")
            plt.show()

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    plt.plot(train_loss, linewidth=2)
    plt.plot(val_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig("FH_v3_train.png")
    plt.show()