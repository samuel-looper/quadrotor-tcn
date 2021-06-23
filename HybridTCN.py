import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from data_loader import TrainSet
from torch.utils.data import DataLoader
from End2EndNet import TConvBlock

# HybridTCN.py: Build and train TCN Hybrid models for quadrotor motion modeling


class HybridTCNComponent(nn.Module):
    def __init__(self, past_state_length, state_size):
        # TCN Component of Hybrid TCN models
        # Input: Time series of past robot state, past control input, and current control input (bs x 16 x (P+1))
        # Output: Predicted intermediate state (bs x state_size x 1)

        super(HybridTCNComponent, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.t = 54     # Size of hidden representation at fully-connected layer
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
        self.tconv5 = TConvBlock(32, 16, K-2, dilations)
        self.bn5 = torch.nn.BatchNorm1d(16)
        self.relu5 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear((self.t + 1) * 16, 128)
        self.relu6 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, state_size)

    def forward(self, input):
        x = self.relu1(self.bn1(self.tconv1(input)))
        x = self.relu2(self.bn2(self.tconv2(x)))
        x = self.relu3(self.bn3(self.tconv3(x)))
        x = self.relu4(self.bn4(self.tconv4(x[:, :, (self.P - self.t):])))
        x = self.relu5(self.bn5(self.tconv5(x)))
        x = torch.flatten(x, 1, 2)
        x = self.relu6(self.fc1(x))
        x = self.fc2(x)
        return x


class HybridTCNComponent_small(nn.Module):
    def __init__(self, past_state_length, state_size):
        # Smaller TCN Component of Hybrid TCN models
        # Input: Time series of past robot state, past control input, and current control input (bs x 16 x (P+1))
        # Output: Predicted intermediate state (bs x state_size x 1)

        super(HybridTCNComponent_small, self).__init__()
        K = 5
        dilations = [1, 2, 4, 8]
        self.P = past_state_length
        self.t = 54     # Size of hidden representation at fully-connected layer
        self.tconv1 = TConvBlock(16, 16, K, dilations)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()
        self.tconv2 = TConvBlock(16, 32, K, dilations)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.relu2 = torch.nn.ReLU()
        self.tconv3 = TConvBlock(32, 32, K, dilations)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.relu3 = torch.nn.ReLU()
        self.tconv4 = TConvBlock(32, 16, K, dilations)
        self.bn4 = torch.nn.BatchNorm1d(16)
        self.relu4 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear((self.t + 1) * 16, 64)
        self.relu5 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, state_size)

    def forward(self, input):
        x = self.relu1(self.bn1(self.tconv1(input)))
        x = self.relu2(self.bn2(self.tconv2(x)))
        x = self.relu3(self.bn3(self.tconv3(x)))
        x = self.relu4(self.bn4(self.tconv4(x[:, :, (self.P - self.t):])))
        x = torch.flatten(x, 1, 2)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x


class HybridTCN(nn.Module):
    def __init__(self, l, m, d, kt, kr, ixx, iyy, izz, past_state_length, device, motor=False, accel_error=False):
        # Hybrid TCN model for quadrotor motion prediction with a TCN component learning motor dynamics
        # Formulated as a dynamic system models such that the forward pass computes the state derivative
        # Input: Time series of past robot state, past control input, and current control input (bs x 16 x (P+1))
        # Output: Predicted quadrotor state derivative (bs x 16 x 1)

        super().__init__()
        self.l = l
        self.m = m
        self.d = d
        self.kt = kt
        self.kr = kr
        self.motor = motor
        self.accel_error = accel_error
        self.I = torch.tensor([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
        if motor and accel_error:
            self.motor_net = HybridTCNComponent_small(past_state_length, 4).to(device)
            self.accel_net = HybridTCNComponent_small(past_state_length, 6).to(device)
        else:
            self.motor_net = HybridTCNComponent_small(past_state_length, 4).to(device) if motor else None
            self.accel_net = HybridTCNComponent_small(past_state_length, 6).to(device) if accel_error else None
        self.torque_mat = torch.tensor([[1, 1, 1, 1],
                          [0.707 * self.l, -0.707 * self.l, -0.707 * self.l, 0.707 * self.l],
                          [-0.707 * self.l, -0.707 * self.l, 0.707 * self.l, 0.707 * self.l],
                          [-self.d, self.d, -self.d, self.d]])
        self.select = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1/self.m, 0, 0, 0]])
        self.g = torch.tensor([[0], [0], [9.8067]])
        # Vectorized thrust calculation coef. matrix from system ID motor model
        self.thrust_mat = torch.tensor([[0.0011, -0.0069, 2.2929],
                                      [-0.0005, -0.0088, 2.5556],
                                      [0.001, -0.0121, 2.2989],
                                      [-0.0001, -0.0116, 2.5572]])

    def forward(self, t, input):
        # Calculates derivative of quadrotor state for dynamic system modeling

        # parse latest state from input
        state = torch.transpose(input[:, :, -2], 0, 1)
        ang = state[0:3, 0]
        rate = state[6:9, :]
        vel = state[9:12, :]
        motor_cmd = state[12:, :]

        # update thrusts
        if self.motor:
            thrusts = torch.transpose(self.motor_net(input), 0, 1)
        else:
            cmd_mat = torch.cat(
                (torch.square(torch.transpose(motor_cmd, 0, 1)), torch.transpose(motor_cmd, 0, 1), torch.ones((1, 4))),
                dim=0)
            thrusts = torch.unsqueeze(torch.diagonal(torch.mm(self.thrust_mat, cmd_mat)), 1)

        torques = torch.mm(self.torque_mat, thrusts)  # update torques

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

        # Calculate orientation derivative
        M = torch.tensor([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])
        m_inv = torch.inverse(M)
        ang_dot = torch.mm(m_inv, rate)

        # Calculate acceleration
        vel_dot = torch.mm(rbi, torch.mm(self.select, torques)) - self.kt * vel - self.g

        # Calculate body rate derivative
        rate_dot = torch.mm(torch.inverse(self.I), torques[1:] - torch.cross(rate, torch.mm(self.I, rate), dim=0)
                             - self.kr * rate)

        # Construct and return output state derivative tensor
        state_dot = torch.transpose(torch.cat([ang_dot, vel, rate_dot, vel_dot, torch.zeros((4, 1))]), 0, 1)
        if self.accel_error:
            state_dot[:, 6:12] += self.accel_net(input)
        output = torch.zeros(input.shape)
        output[:, :, -2] = state_dot
        return output


def train_hybrid(loss, net, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, name):
    # Performs training and validation for Hybrid TCN models in PyTorch
    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=wd)  # Define Adam optimization algorithm
    delta_t = 0.01  # Forward simulation time based on input data
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

            output_gt = label[0, 6:12, 0]                               # Define label as the future truncated state
            feedforward = torch.zeros(label.shape)                      # Add future control input to input state
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)

            optimizer.zero_grad()                                       # Reset gradients
            sim_out = odeint(net, input, torch.tensor([0, delta_t]))    # Forward Simulation
            pred = sim_out[1, 0, 6:12, -2]
            minibatch_loss = loss(pred, output_gt)                      # Compute loss
            minibatch_loss.backward()                                   # Backpropagation
            optimizer.step()                                            # Optimization

            epoch_train_loss += minibatch_loss.item() / train_len
            moving_av += minibatch_loss.item()
            i += 1
            if i % 50 == 0:
                print("Training {}% finished".format(round(100 * i * bs / train_len, 4)))
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

                output_gt = label[:, 6:12, :]                                  # Define label as the future truncated state
                feedforward = torch.zeros(label.shape)                      # Add future control input to input state
                feedforward[:, 12:, :] = label[:, 12:, :]
                input = torch.cat((input, feedforward), 2)

                optimizer.zero_grad()                                       # Reset gradients
                sim_out = odeint(net, input, torch.tensor([0, delta_t]))    # Forward Simulation
                pred = sim_out[1, 0, 6:12, -2]
                minibatch_loss = loss(pred, output_gt)                      # Compute loss

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
    # Simulation Model Parameters
    l = 0.211  # length (m)
    d = 1.7e-5  # blade parameter
    m = 1  # mass (kg)
    kt = 2.35e-14  # translational drag coefficient
    kr = 0.0099  # rotational drag coefficient
    ixx = 0.002  # moment of inertia about X-axis
    iyy = 0.002  # moment of inertia about Y-axis
    izz = 0.001  # moment of inertia about Z-axis

    # Training hyperparameters
    bs = 1
    past_window = 64
    lr = 0.001
    wd = 0.0005
    epochs = 30
    loss = nn.L1Loss()

    # Define training/validation datasets and dataloaders
    tv_set = TrainSet('data/AscTec_Pelican_Flight_Dataset.mat', past_window, 1, full_state=True)
    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("GPU")
    else:
        device = torch.device("cpu")
        print("CPU")

    # Main Training Loop
    model = HybridTCN(l, m, d, kt, kr, ixx, iyy, izz, past_window, device, motor=True, accel_error=False)
    train_hybrid(loss, model, train_loader, val_loader, device, bs, epochs, lr, wd, train_len, val_len, "MotorHybrid")
