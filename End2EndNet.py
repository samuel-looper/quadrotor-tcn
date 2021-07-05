from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchsummary
from data_loader import TrainSet
from torch.utils.data import DataLoader

# End2End.py: Build and train End2EndNet for robotic system modeling
# TCN implementation based on Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN) by  Shaojie Bai
# Link: https://github.com/locuslab/TCN


class Chomp1d(nn.Module):
    # PyTorch module that truncates discrete convolution output for the purposes of causal convolutions
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TConv(nn.Module):
    # Module representing a single causal convolution (truncated 1D convolution)
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TConv, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.init_weights()

    def init_weights(self):  # Initializes weights to positive values
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.net(x)


class TConvBlock(nn.Module):
    # Module representing a temporal convolution block which consists of:
    # - causal convolutions
    # - sequence of conv layers with dilations that increase exponentially

    def __init__(self, c_in, c_out, k, dilations):
        super(TConvBlock, self).__init__()
        self.dsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None  # Downsample layer for residual if required
        self.lookback = 0
        layers = []

        # Adds sequence of causal convolutions to module based on input dilations
        for i in range(len(dilations)):
            d = dilations[i]
            if i == 0:  # Downsample w.r.t channel size at the first convolution
                layers += [TConv(c_in, c_out, k, stride=1, dilation=d, padding=(k - 1) * d)]
            else:
                layers += [TConv(c_out, c_out, k, stride=1, dilation=d, padding=(k - 1) * d)]

            self.lookback += (k - 1) * d    # Calculates total lookback window for layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Model forward pass including residual connection
        out = self.network(x)
        res = x if self.dsample is None else self.dsample(x)
        return out + res


class End2EndNet_3(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_3, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 32, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(32, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.tconv3(x2)
        out = x3[:, :, self.P:]

        return out


class End2EndNet_4(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_4, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 32, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(32, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = self.tconv4(x3)
        out = x4[:, :, self.P:]

        return out


class End2EndNet_5(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_5, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 32, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(32, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.tconv5(x4)
        out = x5[:, :, self.P:]

        return out


class End2EndNet_6(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_6, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = self.tconv7(x5)
        out = x6[:, :, self.P:]

        return out


class End2EndNet_8(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_8, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 16, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(16, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 32, K, dilations)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(32, 32, K, dilations)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6)))
        x8 = self.tconv8(x7)
        out = x8[:, :, self.P:]

        return out


class End2EndNet_10(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_10, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 16, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(16, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 32, K, dilations)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(32, 32, K, dilations)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(32, 32, K, dilations)
        self.bn8 = torch.nn.BatchNorm1d(32)
        self.relu8 = torch.nn.ReLU()
        self.tconv9 = TConvBlock(32, 32, K, dilations)
        self.bn9 = torch.nn.BatchNorm1d(32)
        self.relu9 = torch.nn.ReLU()
        self.tconv10 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6)))
        x8 = x7 + self.relu8(self.bn8(self.tconv8(x7)))
        x9 = self.relu9(self.bn9(self.tconv9(x8)))
        x10 = self.tconv10(x9)
        out = x10[:, :, self.P:]

        return out


class End2EndNet_12(nn.Module):
    def __init__(self, past_state_length, future_state_length):
        # Final End2EndNet design with fewer layers, fewer channels, no dropout,
        # and control inputs at the front of the network

        # Input: Time series of past robot state, past control input, and future control input (bs x 16 x (P+F))
        # Output: Time series of future truncated robot state (bs x 6 x F)

        super(End2EndNet_12, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.F = future_state_length

        self.tconv1 = TConvBlock(16, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 16, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(16, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 32, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.tconv5 = TConvBlock(32, 32, K, dilations)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.relu5 = torch.nn.ReLU()
        self.tconv6 = TConvBlock(32, 32, K, dilations)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.relu6 = torch.nn.ReLU()
        self.tconv7 = TConvBlock(32, 32, K, dilations)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.relu7 = torch.nn.ReLU()
        self.tconv8 = TConvBlock(32, 32, K, dilations)
        self.bn8 = torch.nn.BatchNorm1d(32)
        self.relu8 = torch.nn.ReLU()
        self.tconv9 = TConvBlock(32, 32, K, dilations)
        self.bn9 = torch.nn.BatchNorm1d(32)
        self.relu9 = torch.nn.ReLU()
        self.tconv10 = TConvBlock(32, 32, K, dilations)
        self.bn10 = torch.nn.BatchNorm1d(32)
        self.relu10 = torch.nn.ReLU()
        self.tconv11 = TConvBlock(32, 32, K, dilations)
        self.bn11 = torch.nn.BatchNorm1d(32)
        self.relu11 = torch.nn.ReLU()
        self.tconv12 = TConvBlock(32, 6, K, dilations)

    def forward(self, input):
        x1 = self.relu1(self.bn1(self.tconv1(input)))
        x2 = x1 + self.relu2(self.bn2(self.tconv2(x1)))
        x3 = self.relu3(self.bn3(self.tconv3(x2)))
        x4 = x3 + self.relu4(self.bn4(self.tconv4(x3)))
        x5 = self.relu5(self.bn5(self.tconv5(x4)))
        x6 = x5 + self.relu6(self.bn6(self.tconv6(x5)))
        x7 = self.relu7(self.bn7(self.tconv7(x6)))
        x8 = x7 + self.relu8(self.bn8(self.tconv8(x7)))
        x9 = self.relu9(self.bn9(self.tconv9(x8)))
        x10 = x9 + self.relu10(self.bn10(self.tconv10(x9)))
        x11 = self.relu11(self.bn11(self.tconv11(x10)))
        x12 = self.tconv12(x11)
        out = x12[:, :, self.P:]

        return out


class WeightedTemporalLoss(nn.Module):
    # Custom loss function that applies mean square error, with a decaying weight term
    def __init__(self, weight=None, size_average=True):
        super(WeightedTemporalLoss, self).__init__()

    def forward(self, prediction, label):
        error = prediction-label
        error = torch.mean(error**2, (0, 1))
        weights = torch.flip(torch.arange(0.88, 1, 0.002), [0])
        return torch.mean(torch.mul(error, weights))


def train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, name):
    # Performs training and validation for End2EndNet in PyTorch

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

        # Training
        for data in train_loader:
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2).to(device)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2).to(device)  # Load labels

            output = label[:, 6:12, :]                  # Define label as the future truncated state
            feedforward = torch.zeros(label.shape)      # Add future control input to input state
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)
            
            optimizer.zero_grad()                       # Reset gradients
            pred = net(input)                           # Forward Pass
            minibatch_loss = loss(pred, output)         # Compute loss
            epoch_train_loss += minibatch_loss.item() / train_len
            moving_av += minibatch_loss.item()

            minibatch_loss.backward()                   # Backpropagation
            optimizer.step()                            # Optimization
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
                input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2).to(device)  # Load Input data
                label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2).to(device)  # Load labels

                output = label[:, 6:12, :]                  # Define label as the future truncated state
                feedforward = torch.zeros(label.shape)      # Add future control input to input state
                feedforward[:, 12:, :] = label[:, 12:, :]
                input = torch.cat((input, feedforward), 2)

                optimizer.zero_grad()                       # Reset gradients
                pred = net(input)                           # Forward Pass
                minibatch_loss = loss(pred, output)         # Compute loss
                epoch_val_loss += minibatch_loss.item() / val_len
                i += 1
                if i % 100 == 0:
                    print(i)

            val_loss.append(epoch_val_loss)
            print("Validation Loss: {}".format(epoch_val_loss))
            if best_epoch == 0 or epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(net.state_dict(), "{}.pth".format(name))

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    fig, ax = plt.subplots()
    ax.plot(train_loss, linewidth=2)
    ax.plot(val_loss, linewidth=2)
    ax.set_title("{} Training & Validation Losses".format(name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(["Training Loss", "Validation Loss"])
    fig.savefig("{}.png".format(name))
    fig.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("GPU")
    else:
        device = torch.device("cpu")
        print("CPU")

    lr = 0.0001
    wd = 0.00005
    epochs = 50
    bs = 16
    P = 1
    F = 90

    loss = torch.nn.L1Loss()  # Define L1 Loss

    # Define training/validation datasets and dataloaders
    tv_set = TrainSet('data/AscTec_Pelican_Flight_Dataset.mat', P, F, full_state=True)
    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    # Run main training loop
    net = End2EndNet_3(P, F).to(device)
    torchsummary.summary(net,  (16, P+F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_3layer")

    net = End2EndNet_4(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_4layer")

    net = End2EndNet_5(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_5layer")

    net = End2EndNet_6(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_6layer")

    net = End2EndNet_8(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_8layer")

    net = End2EndNet_10(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_10layer")

    net = End2EndNet_12(P, F).to(device)
    torchsummary.summary(net, (16, P + F))
    train_model(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "End2End_12layer")

