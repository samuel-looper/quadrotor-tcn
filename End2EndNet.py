from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchsummary
from data_loader import TrainSet
from torch.utils.data import DataLoader
import math
PATH = 'E2E_v5.pth'


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
        self.tconv2 = TConvBlock(L + P, 16, 16, K, d)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(L + P, 16, 32, K, d)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(P + int(L/2), 32, 32, K, d)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(P + int(L/2), 32, 64, K, d)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(P + int(L/2), 64, 64, K, d)
        self.bn6 = torch.nn.BatchNorm1d(64)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(P, 64, 128, K, d)
        self.bn7 = torch.nn.BatchNorm1d(128)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(P, 128, 128, K, d)
        self.bn8 = torch.nn.BatchNorm1d(128)
        self.relu8 = torch.nn.ReLU()
        self.tconv9 = TConvBlock(P, 128, 256, K, d)
        self.bn9 = torch.nn.BatchNorm1d(256)
        self.relu9 = torch.nn.ReLU()
        self.tconv10 = TConvBlock(P, 256, 6, K, d)

    def forward(self, input):
        # Assume X: batch by length by channel size
        # print(input.shape)
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3[:, :, int(self.L/2):] + self.relu4(self.bn4(self.tconv4(x3[:, :, int(self.L/2):])))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6[:, :, int(self.L/2):])))
        x8 = x7 + self.relu8(self.bn8(self.tconv8(x7)))
        x9 = self.relu9(self.bn9(self.tconv9(x8)))
        x10 = self.tconv10(x9)
        # print(x.shape)
        return x10


def train_model():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")

    lr = 0.0001
    wd = 0.00005
    epochs = 100
    bs = 16
    L =64
    P = 60
    tv_set = TrainSet('data/AscTec_Pelican_Flight_Dataset.mat', L, P, full_set=True)

    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    net = E2ESingleStepTCN(L, P)#.to(device)
    torchsummary.summary(net, (16, 124))

    loss = torch.nn.MSELoss()  # Define Mean Square Error Loss
    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=wd)  # Define Adam optimization algorithm

    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0

    print("Training Length: {}".format(int(train_len / bs)))
    for epoch in range(1, epochs + 1):
        print("Epoch # {}".format(epoch))
        net.train(True)
        epoch_train_loss = 0
        moving_av = 0
        i = 0

        for data in train_loader:
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)#.to(device)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)#.to(device)  # Load labels

            output = label[:, 6:12, :]
            feedforward = torch.zeros(label.shape)
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)
            
            optimizer.zero_grad()  # Reset gradients
            pred = net(input)  # Forward Pass
            minibatch_loss = loss(pred, output)  # Compute loss
            epoch_train_loss += minibatch_loss.item() / train_len
            moving_av += minibatch_loss.item()

            minibatch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimization
            i += 1
            if i % 50 == 0:
                print("Training {}% finished".format(round(100 * i* bs / train_len, 4)))
                print(moving_av / 50)
                moving_av = 0

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
                input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)#.to(device)  # Load Input data
                label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)#.to(device)  # Load labels

                output = label[:, 6:12, :]
                feedforward = torch.zeros(label.shape)
                feedforward[:, 12:, :] = label[:, 12:, :]
                input = torch.cat((input, feedforward), 2)

                optimizer.zero_grad()  # Reset gradients
                pred = net(input)  # Forward Pass
                minibatch_loss = loss(pred, output)  # Compute loss
                epoch_val_loss += minibatch_loss.item() / val_len
                i += 1
                if i % 100 == 0:
                    print(i)

            val_loss.append(epoch_val_loss)
            print("Validation Loss: {}".format(epoch_val_loss))
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
            plt.savefig("E2E_v5_train_intermediate.png")
            plt.show()

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    plt.plot(train_loss, linewidth=2)
    plt.plot(val_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig("E2E_v5_train.png")
    plt.show()


if __name__ == "__main__":
    train_model()
