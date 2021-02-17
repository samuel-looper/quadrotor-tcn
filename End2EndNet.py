import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchsummary
from data_loader import SinglePredDatasetTrain
from data_loader import SinglePredDatasetTest
from torch.utils.data import DataLoader
import math
PATH = 'E2E_1Step_best.pth'


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return (out + res)


class TConvBlock(nn.Module):
    # Temporal Convolution block that accepts an input of Lxc_in with a dilation factor of d and performs
    # causal convolution on the input with a kernel size of K to return an output size Lxc_out

    # Note that the look-back length is not necessarily L but is actually the nearest value K*d^i < L for some int i
    def __init__(self, L, c_in, c_out, K, d):
        super(TConvBlock, self).__init__()
        layers = []
        n = math.floor(math.log(L / K, d))
        for i in range(n):
            if i == 0:
                layers += [TConvLayer(c_in, c_out, K, stride=1, dilation=d, padding=(K - 1) * d)]
            else:
                layers += [TConvLayer(c_out, c_out, K, stride=1, dilation=d, padding=(K - 1) * d)]


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class E2ESingleStepTCN(nn.Module):
    def __init__(self, lookback, forward_step):
        super(E2ESingleStepTCN, self).__init__()
        L = lookback
        P = forward_step
        K = 8
        d = 2
        self.L = L
        self.tconv1 = TConvBlock(L+P, 16, 16, K, d)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(L+P, 16, 32, K, d)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(P + int(L/2), 32, 32, K, d)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(P + int(L/2), 32, 64, K, d)
        self.bn4 = torch.nn.BatchNorm1d(64)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(P, 64, 64, K, d)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(P, 64, 6, K, d)



    def forward(self, input):
        # Assume X: batch by length by channel size

        x = self.tconv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.relu3(self.bn3(self.tconv3(x[:, :, int(self.L/2):])))
        x = self.relu4(self.bn4(self.tconv4(x)))
        x = self.relu5(self.bn5(self.tconv5(x[:, :, int(self.L/2):])))
        x = self.tconv6(x)
        return x

# In construction
# class QuadrotorDynamics(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rates_net = E2ESingleStepTCN()
#
#     def forward(self, t, input):
#         state = torch.transpose(input[:, :, -2], 0, 1)
#         ang = state[0:3, 0]
#         rate = state[6:9, :]
#         vel = state[9:12, :]
#
#         rates = self.rates_net(input)
#         rate_dot = rates[:3, :]
#         vel_dot = rates[3:, :]
#
#         # Calculate rotation matrix
#         s_phi = (torch.sin(ang[0])).item()
#         c_phi = (torch.cos(ang[0])).item()
#         c_theta = (torch.cos(ang[1])).item()
#
#         M = torch.tensor([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])
#
#         m_inv = torch.inverse(M)
#         ang_dot = torch.mm(m_inv, rate)
#
#         state_dot = torch.transpose(torch.cat([ang_dot, vel, rate_dot, vel_dot, torch.zeros((4, 1))]), 0, 1)
#         output = torch.zeros(input.shape)
#         output[:, :, -2] = state_dot
#         return output

def train_model():
    lr = 0.001
    wd = 0.0005
    epochs = 30
    bs = 16
    L =64
    P = 20
    tv_set = SinglePredDatasetTrain('data/AscTec_Pelican_Flight_Dataset.mat', L, P, full_set=True)
    # datapoint = test_set[426]

    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    net = E2ESingleStepTCN(L, P)
    # torchsummary.summary(net, (16, 84))

    loss = torch.nn.MSELoss()  # Define Mean Square Error Loss
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)  # Define Adam optimization algorithm

    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0

    print("Training Length: {}".format(int(train_len / bs)))
    for epoch in range(1, epochs + 1):
        # Training
        print("Training")
        net.train(True)
        epoch_train_loss = 0
        i = 0

        for data in train_loader:
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels

            output = label[:, 6:12, :]
            feedforward = torch.zeros(label.shape)
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)
            
            optimizer.zero_grad()  # Reset gradients
            pred = net(input)  # Forward Pass
            minibatch_loss = loss(pred, output)  # Compute loss
            epoch_train_loss += minibatch_loss.item() / train_len

            minibatch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimization
            i += 1
            if i % 200 == 0:
                print(i)
                print(minibatch_loss)

        train_loss.append(epoch_train_loss)
        print("Training Error for this Epoch: {}".format(epoch_train_loss))

        # Validation
        print("Validation")
        net.train(False)
        net.eval()
        epoch_val_loss = 0
        i = 0
        with torch.no_grad():
            for data in val_loader:
                input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
                label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels

                output = label[:, 6:12, :]
                feedforward = torch.zeros(label.shape)
                feedforward[:, 12:, :] = label[:, 12:, :]
                input = torch.cat((input, feedforward), 2)

                optimizer.zero_grad()  # Reset gradients
                pred = net(input)  # Forward Pass
                minibatch_loss = loss(pred, output)  # Compute loss
                epoch_val_loss += minibatch_loss.item() / val_len
                i += 1
                if i % 40 == 0:
                    print(i)

            val_loss.append(epoch_val_loss)
            print(epoch_val_loss)
            if best_epoch == 0 or epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(net.state_dict(), PATH)

            # Plotting
            plt.plot(train_loss, linewidth=2)
            plt.plot(val_loss, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend(["Training Loss", "Validation Loss"])
            plt.savefig("E2E_1Step_losses_intermediate.png")
            plt.show()

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    plt.plot(train_loss, linewidth=2)
    plt.plot(val_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig("E2E_1Step_losses.png")
    plt.show()

def test_model():
    bs = 1
    test_set = SinglePredDatasetTest('data/AscTec_Pelican_Flight_Dataset.mat', 64, 1)
    # datapoint = test_set[426]

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    net = E2ESingleStepTCN()
    net.load_state_dict(torch.load(PATH))
    net.train(False)
    net.eval()

    loss = torch.nn.MSELoss()  # Define Mean Square Error Loss

    test_loss = []
    i=0
    with torch.no_grad():
        for data in test_loader:
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels
            output = label[:, :6, 0]
            feedforward = torch.zeros(label.shape)
            feedforward[:, 6:, :] = label[:, 6:, :]
            input = torch.cat((input, feedforward), 2)
            pred = net(input)  # Forward Pass
            minibatch_loss = loss(pred, output)  # Compute loss
            test_loss.append(minibatch_loss)
            i += 1
            if i % 40 == 0:
                print(i)
    np.savetxt("test_loss.csv", test_loss)

if __name__ == "__main__":
    train_model()
    # test_model()